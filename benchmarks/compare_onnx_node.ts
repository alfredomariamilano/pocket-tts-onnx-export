import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";

import * as ort from "onnxruntime-node";
import { Tokenizer } from "@huggingface/tokenizers";

const ROOT = process.cwd();
const HF_DIR = path.join(ROOT, "hf");
const ONNX_DIR = path.join(HF_DIR, "onnx");
const OUTPUT_DIR = path.join(ROOT, "comparison_outputs");
const SAMPLE_RATE = 24_000;
const LATENT_DIM = 32;
const AUDIO_CONDITIONING_DIM = 1024;
const TEXT_DIM = 1024;
const DEFAULT_PRECISION = "fp32";
const DEFAULT_BUILTIN_VOICE = "marius";
const DEFAULT_FLOW_STEPS = 10;
const DEFAULT_TEMPERATURE = 0.7;
const CLONE_PROMPT_PATH = path.join(OUTPUT_DIR, "clone_prompt.wav");
const BUILTIN_OUTPUT_PATH = path.join(OUTPUT_DIR, "node_builtin.wav");
const CLONE_OUTPUT_PATH = path.join(OUTPUT_DIR, "node_voice_clone.wav");
const REPORT_PATH = path.join(OUTPUT_DIR, "node_onnx_report.json");
const DEFAULT_TEXT =
  "This is a direct quality comparison between the ONNX Node runtime and the native Python Pocket TTS runtime.";
const DEBUG_COMPARE = process.env.DEBUG_COMPARE === "1";
const MIMI_STEPS_PER_LATENT = 16;
type Precision = "fp32" | "int8" | "q4";
type TensorData = Float32Array | BigInt64Array | Uint8Array;

type ModelStats = {
  mountMs: number;
  calls: number;
  totalInferenceMs: number;
};

type ScenarioStats = {
  totalMs: number;
  audioSeconds: number;
  models: Record<string, { calls: number; totalInferenceMs: number; averageInferenceMs: number }>;
  outputPath: string;
  outputPathFloat32: string;
  rawPath: string;
  audioStats: {
    min: number;
    max: number;
    rms: number;
  };
};

type VoiceStateTensor = {
  name: string;
  dtype: "float32" | "int64" | "bool";
  shape: number[];
  data: TensorData;
};

type VoiceState = Map<string, VoiceStateTensor>;

type VoiceStateBundleTensor = {
  name: string;
  dtype: "float32" | "int64" | "bool";
  shape: number[];
  offset: number;
  nbytes: number;
};

type VoiceStateBundle = {
  tensors: VoiceStateBundleTensor[];
  totalBytes: number;
  data: Buffer;
};

type AudioConditioning = {
  embeddings: Float32Array;
  frames: number;
};

type ScenarioConditioning = { kind: "flow_state"; value: VoiceStateSource };

type VoiceStateSource =
  | { kind: "serialized"; jsonPath: string }
  | { kind: "runtime"; bundle: VoiceStateBundle };

type FlowStateFileMeta = {
  format: "pocket_tts_flow_state_v2";
  tensor_count: number;
  total_bytes: number;
  tensors: Array<{
    name: string;
    source_key: string;
    dtype: "float32" | "int64" | "bool";
    shape: number[];
    offset: number;
    nbytes: number;
  }>;
};

type WavData = {
  sampleRate: number;
  channels: number;
  samples: Float32Array;
};

const modelStats: Record<string, ModelStats> = {};

function ensureDir(dirPath: string): void {
  fs.mkdirSync(dirPath, { recursive: true });
}

function suffixForPrecision(precision: Precision): string {
  if (precision === "fp32") {
    return ".onnx";
  }
  return `_${precision}.onnx`;
}

function modelPath(baseName: string, precision: Precision): string {
  return path.join(ONNX_DIR, `${baseName}${suffixForPrecision(precision)}`);
}

async function mountSession(modelName: string, filePath: string): Promise<ort.InferenceSession> {
  const start = performance.now();
  const session = await ort.InferenceSession.create(filePath);
  const mountMs = performance.now() - start;
  modelStats[modelName] = {
    mountMs,
    calls: 0,
    totalInferenceMs: 0,
  };
  return session;
}

async function timedRun(
  modelName: string,
  session: ort.InferenceSession,
  feeds: Record<string, ort.Tensor>,
): Promise<Record<string, ort.Tensor>> {
  const start = performance.now();
  const result = await session.run(feeds);
  const elapsed = performance.now() - start;
  modelStats[modelName].calls += 1;
  modelStats[modelName].totalInferenceMs += elapsed;
  return result;
}

function normalRandom(): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function makeNoiseTensor(std: number): Float32Array {
  const values = new Float32Array(LATENT_DIM);
  for (let i = 0; i < values.length; i += 1) {
    values[i] = normalRandom() * std;
  }
  return values;
}

function prepareTextPrompt(text: string): { prepared: string; framesAfterEos: number } {
  let prepared = text.trim();
  if (prepared.length === 0) {
    throw new Error("Text prompt cannot be empty");
  }
  prepared = prepared.replace(/\r/g, " ").replace(/\n/g, " ").replace(/ {2,}/g, " ");
  const words = prepared.split(/\s+/).filter(Boolean);
  const framesAfterEosGuess = words.length <= 4 ? 3 : 1;
  if (prepared[0] !== prepared[0].toUpperCase()) {
    prepared = prepared[0].toUpperCase() + prepared.slice(1);
  }
  if (/\p{L}|\p{N}/u.test(prepared[prepared.length - 1] ?? "")) {
    prepared += ".";
  }
  if (prepared.split(/\s+/).length < 5) {
    prepared = "        " + prepared;
  }
  return { prepared, framesAfterEos: framesAfterEosGuess + 2 };
}

function estimateMaxGenLen(tokenCount: number): number {
  const genSeconds = tokenCount / 3.0 + 2.0;
  return Math.ceil(genSeconds * 12.5);
}

function tensorFromFloat(data: Float32Array, dims: number[]): ort.Tensor {
  return new ort.Tensor("float32", data, dims);
}

function tensorFromInt64(data: BigInt64Array, dims: number[]): ort.Tensor {
  return new ort.Tensor("int64", data, dims);
}

function tensorFromBool(data: Uint8Array, dims: number[]): ort.Tensor {
  return new ort.Tensor("bool", data, dims);
}

function createFloatState(dims: number[], fillNaN: boolean): Float32Array {
  const total = dims.reduce((acc, value) => acc * value, 1);
  const values = new Float32Array(total);
  if (fillNaN) {
    values.fill(Number.NaN);
  }
  return values;
}

function getInputMeta(
  session: ort.InferenceSession,
  inputName: string,
): { type: string; shape?: ReadonlyArray<number | string>; dimensions?: ReadonlyArray<number | string> } | undefined {
  const rawMetadata = session.inputMetadata as unknown as
    | Record<string, { type: string; shape?: ReadonlyArray<number | string>; dimensions?: ReadonlyArray<number | string> }>
    | Array<{ name: string; type: string; shape?: ReadonlyArray<number | string>; dimensions?: ReadonlyArray<number | string> }>;
  if (Array.isArray(rawMetadata)) {
    return rawMetadata.find((entry) => entry.name === inputName);
  }
  return rawMetadata[inputName];
}

function normalizeTensorType(type: string): "float32" | "int64" | "bool" | "unknown" {
  if (type === "float32" || type === "tensor(float)") {
    return "float32";
  }
  if (type === "int64" || type === "tensor(int64)") {
    return "int64";
  }
  if (type === "bool" || type === "tensor(bool)") {
    return "bool";
  }
  return "unknown";
}

function metadataShape(session: ort.InferenceSession, inputName: string): number[] {
  const meta = getInputMeta(session, inputName);
  const dims = meta?.dimensions ?? meta?.shape;
  if (!dims) {
    throw new Error(`Missing input metadata for ${inputName}`);
  }
  return dims.map((dim) => (typeof dim === "number" ? dim : 0));
}

function initState(
  session: ort.InferenceSession,
): VoiceState {
  const state = new Map<string, VoiceStateTensor>();
  for (const inputName of session.inputNames.filter((name: string) => name.startsWith("state_"))) {
    const meta = getInputMeta(session, inputName);
    if (!meta) {
      throw new Error(`Missing input metadata for ${inputName}`);
    }
    const shape = metadataShape(session, inputName);
    const fillNaN = false;
    const tensorType = normalizeTensorType(meta.type);
    if (tensorType === "float32") {
      state.set(inputName, {
        name: inputName,
        dtype: "float32",
        shape,
        data: createFloatState(shape, fillNaN),
      });
    } else if (tensorType === "int64") {
      state.set(inputName, {
        name: inputName,
        dtype: "int64",
        shape,
        data: new BigInt64Array(shape.reduce((acc, value) => acc * value, 1)),
      });
    } else if (tensorType === "bool") {
      state.set(inputName, {
        name: inputName,
        dtype: "bool",
        shape,
        data: new Uint8Array(shape.reduce((acc, value) => acc * value, 1)),
      });
    } else {
      throw new Error(`Unsupported state type ${meta.type} for ${inputName}`);
    }
  }
  return state;
}

function initFlowState(flowMainSession: ort.InferenceSession, cacheLength: number): VoiceState {
  const state = new Map<string, VoiceStateTensor>();
  for (const inputName of flowMainSession.inputNames.filter((name: string) => name.startsWith("state_"))) {
    const meta = getInputMeta(flowMainSession, inputName);
    if (!meta) {
      throw new Error(`Missing input metadata for ${inputName}`);
    }
    const shape = metadataShape(flowMainSession, inputName);
    const tensorType = normalizeTensorType(meta.type);
    if (tensorType === "float32") {
      const targetShape = shape.length === 5
        ? [shape[0], shape[1], shape[2] > 0 ? shape[2] : cacheLength, shape[3], shape[4]]
        : shape;
      state.set(inputName, {
        name: inputName,
        dtype: "float32",
        shape: targetShape,
        data: createFloatState(targetShape, targetShape.length === 5),
      });
    } else if (tensorType === "int64") {
      state.set(inputName, {
        name: inputName,
        dtype: "int64",
        shape,
        data: new BigInt64Array(shape.reduce((acc, value) => acc * value, 1)),
      });
    } else if (tensorType === "bool") {
      state.set(inputName, {
        name: inputName,
        dtype: "bool",
        shape,
        data: new Uint8Array(shape.reduce((acc, value) => acc * value, 1)),
      });
    } else {
      throw new Error(`Unsupported state type ${meta.type} for ${inputName}`);
    }
  }
  return state;
}

function initMimiState(mimiDecoderSession: ort.InferenceSession): VoiceState {
  return initState(mimiDecoderSession);
}

function readJson<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function loadAudioConditioning(basePath: string): AudioConditioning {
  const meta = readJson<{ shape: number[]; dtype: "float32" }>(`${basePath}.json`);
  if (meta.dtype !== "float32") {
    throw new Error(`Unsupported conditioning dtype ${meta.dtype} for ${basePath}.json`);
  }
  const bin = fs.readFileSync(`${basePath}.bin`);
  const embeddings = new Float32Array(bin.buffer, bin.byteOffset, bin.byteLength / Float32Array.BYTES_PER_ELEMENT);
  const [batch, frames, dim] = meta.shape;
  if (batch !== 1 || dim !== AUDIO_CONDITIONING_DIM) {
    throw new Error(`Unexpected conditioning shape ${meta.shape.join("x")} for ${basePath}.json`);
  }
  return { embeddings: new Float32Array(embeddings), frames };
}

function readTypedSlice(blob: Buffer, offset: number, nbytes: number, dtype: "float32" | "int64" | "bool"): TensorData {
  const slice = blob.subarray(offset, offset + nbytes);
  if (dtype === "float32") {
    return new Float32Array(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength));
  }
  if (dtype === "int64") {
    return new BigInt64Array(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength));
  }
  return new Uint8Array(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength));
}

function expandFloatCacheToModelShape(compact: Float32Array, compactShape: number[], targetShape: number[]): Float32Array {
  const expanded = createFloatState(targetShape, false);
  const outer = compactShape[0] * compactShape[1];
  const compactT = compactShape[2];
  const targetT = targetShape[2];
  const inner = compactShape.slice(3).reduce((acc, value) => acc * value, 1);
  for (let outerIndex = 0; outerIndex < outer; outerIndex += 1) {
    for (let timeIndex = 0; timeIndex < compactT; timeIndex += 1) {
      const compactOffset = (outerIndex * compactT + timeIndex) * inner;
      const targetOffset = (outerIndex * targetT + timeIndex) * inner;
      expanded.set(compact.subarray(compactOffset, compactOffset + inner), targetOffset);
    }
  }
  return expanded;
}

function readFlowStateCurrentEnd(jsonPath: string): number {
  const meta = readJson<FlowStateFileMeta>(jsonPath);
  const currentEndTensor = meta.tensors.find((entry) => entry.source_key === "current_end")
    ?? meta.tensors.find((entry) => entry.name === "state_1");
  if (!currentEndTensor || currentEndTensor.dtype !== "int64" || currentEndTensor.shape.reduce((acc, value) => acc * value, 1) !== 1) {
    throw new Error(`Could not locate current_end tensor in ${jsonPath}`);
  }
  const binPath = jsonPath.replace(/\.json$/i, ".bin");
  const blob = fs.readFileSync(binPath);
  const value = readTypedSlice(blob, currentEndTensor.offset, currentEndTensor.nbytes, "int64") as BigInt64Array;
  return Number(value[0]);
}

function loadFlowStateV2(jsonPath: string, flowMainSession: ort.InferenceSession, targetCacheLength: number): VoiceState {
  const meta = readJson<FlowStateFileMeta>(jsonPath);
  if (meta.format !== "pocket_tts_flow_state_v2") {
    throw new Error(`Unsupported flow state format in ${jsonPath}`);
  }
  const binPath = jsonPath.replace(/\.json$/i, ".bin");
  const blob = fs.readFileSync(binPath);
  const state = new Map<string, VoiceStateTensor>();
  for (const entry of meta.tensors) {
    const modelShape = metadataShape(flowMainSession, entry.name);
    let data = readTypedSlice(blob, entry.offset, entry.nbytes, entry.dtype);
    let shape = [...entry.shape];
    if (entry.dtype === "float32" && shape.length === 5) {
      const resolvedCacheLength = modelShape[2] > 0 ? modelShape[2] : targetCacheLength;
      const targetShape = [shape[0], shape[1], resolvedCacheLength, shape[3], shape[4]];
      if (targetCacheLength < shape[2]) {
        throw new Error(`Target FlowLM cache length ${targetCacheLength} is smaller than serialized cache length ${shape[2]} for ${entry.name}`);
      }
      if (shape[2] !== targetShape[2]) {
        data = expandFloatCacheToModelShape(data as Float32Array, shape, targetShape);
        shape = targetShape;
      }
    } else if (entry.dtype === "float32" && shape.length === 5 && modelShape[2] > 0 && shape[2] !== modelShape[2]) {
      data = expandFloatCacheToModelShape(data as Float32Array, shape, modelShape);
      shape = modelShape;
    }
    state.set(entry.name, {
      name: entry.name,
      dtype: entry.dtype,
      shape,
      data,
    });
  }
  return state;
}

function cloneState(state: VoiceState): VoiceState {
  const cloned = new Map<string, VoiceStateTensor>();
  for (const [name, tensorInfo] of state.entries()) {
    if (tensorInfo.dtype === "float32") {
      cloned.set(name, { ...tensorInfo, data: new Float32Array(tensorInfo.data as Float32Array) });
    } else if (tensorInfo.dtype === "int64") {
      cloned.set(name, { ...tensorInfo, data: new BigInt64Array(tensorInfo.data as BigInt64Array) });
    } else {
      cloned.set(name, { ...tensorInfo, data: new Uint8Array(tensorInfo.data as Uint8Array) });
    }
  }
  return cloned;
}

function product(dims: readonly number[], start = 0, end = dims.length): number {
  let total = 1;
  for (let index = start; index < end; index += 1) {
    total *= Math.max(1, dims[index] ?? 1);
  }
  return total;
}

function buildVoiceStateBundleFromState(state: VoiceState): VoiceStateBundle {
  const names = [...state.keys()].sort((left, right) => Number(left.replace("state_", "")) - Number(right.replace("state_", "")));
  const tensors: VoiceStateBundleTensor[] = [];
  const chunks: Buffer[] = [];
  let offset = 0;

  for (const name of names) {
    const tensor = state.get(name);
    if (!tensor) {
      continue;
    }
    const raw = Buffer.from(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
    tensors.push({
      name,
      dtype: tensor.dtype,
      shape: [...tensor.shape],
      offset,
      nbytes: raw.byteLength,
    });
    chunks.push(raw);
    offset += raw.byteLength;
  }

  return {
    tensors,
    totalBytes: offset,
    data: Buffer.concat(chunks),
  };
}

function tensorFromBundle(bundle: VoiceStateBundle, descriptor: VoiceStateBundleTensor): ort.Tensor {
  const slice = bundle.data.subarray(descriptor.offset, descriptor.offset + descriptor.nbytes);
  const arrayBuffer = slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
  if (descriptor.dtype === "float32") {
    return new ort.Tensor("float32", new Float32Array(arrayBuffer), descriptor.shape);
  }
  if (descriptor.dtype === "int64") {
    return new ort.Tensor("int64", new BigInt64Array(arrayBuffer), descriptor.shape);
  }
  return new ort.Tensor("bool", new Uint8Array(arrayBuffer), descriptor.shape);
}

function coerceStateTensorToTemplate(source: ort.Tensor, template: VoiceStateTensor): VoiceStateTensor {
  if (source.type === template.dtype && source.dims.length === template.shape.length && source.dims.every((value, index) => value === template.shape[index])) {
    if (source.type === "float32") {
      return { name: template.name, dtype: "float32", shape: [...source.dims], data: new Float32Array(source.data as Float32Array) };
    }
    if (source.type === "int64") {
      return { name: template.name, dtype: "int64", shape: [...source.dims], data: new BigInt64Array(source.data as BigInt64Array) };
    }
    return { name: template.name, dtype: "bool", shape: [...source.dims], data: new Uint8Array(source.data as Uint8Array) };
  }

  if (
    source.type === "float32" &&
    template.dtype === "float32" &&
    source.dims.length === template.shape.length &&
    source.dims.length >= 3 &&
    source.dims[2] < template.shape[2] &&
    source.dims.every((value, index) => index === 2 || value === template.shape[index])
  ) {
    const sourceData = source.data as Float32Array;
    const targetData = new Float32Array(template.data as Float32Array);
    const before = product(template.shape, 0, 2);
    const after = product(template.shape, 3);
    const sourceAxis2 = source.dims[2];
    const targetAxis2 = template.shape[2];
    for (let block = 0; block < before; block += 1) {
      const sourceOffset = block * sourceAxis2 * after;
      const targetOffset = block * targetAxis2 * after;
      targetData.set(sourceData.subarray(sourceOffset, sourceOffset + sourceAxis2 * after), targetOffset);
    }
    return { name: template.name, dtype: "float32", shape: [...template.shape], data: targetData };
  }

  return { ...template, shape: [...template.shape], data: template.dtype === "float32" ? new Float32Array(template.data as Float32Array) : template.dtype === "int64" ? new BigInt64Array(template.data as BigInt64Array) : new Uint8Array(template.data as Uint8Array) };
}

function readCurrentEndFromBundle(bundle: VoiceStateBundle): number {
  let currentEnd = 0;
  for (const descriptor of bundle.tensors) {
    if (descriptor.dtype !== "int64" || descriptor.shape.length !== 1 || descriptor.shape[0] !== 1) {
      continue;
    }
    const tensor = tensorFromBundle(bundle, descriptor);
    const value = Number((tensor.data as BigInt64Array)[0] ?? 0n);
    if (Number.isFinite(value) && value > currentEnd) {
      currentEnd = value;
    }
  }
  return currentEnd;
}

function buildSeededFlowState(flowMainSession: ort.InferenceSession, bundle: VoiceStateBundle, requiredCacheLength?: number): VoiceState {
  const cacheLengths = bundle.tensors.filter((tensor) => tensor.dtype === "float32" && tensor.shape.length === 5).map((tensor) => tensor.shape[2]);
  const compactCacheLength = cacheLengths.length ? Math.max(...cacheLengths) : 1;
  const targetCacheLength = Math.max(compactCacheLength, requiredCacheLength ?? 1);
  const state = initFlowState(flowMainSession, targetCacheLength);
  for (const descriptor of bundle.tensors.sort((left, right) => Number(left.name.replace("state_", "")) - Number(right.name.replace("state_", "")))) {
    const template = state.get(descriptor.name);
    if (!template) {
      continue;
    }
    const seededTensor = coerceStateTensorToTemplate(tensorFromBundle(bundle, descriptor), template);
    state.set(descriptor.name, seededTensor);
  }
  return state;
}

function loadVoiceStateBundleV2(jsonPath: string): VoiceStateBundle {
  const meta = readJson<FlowStateFileMeta>(jsonPath);
  if (meta.format !== "pocket_tts_flow_state_v2") {
    throw new Error(`Unsupported flow state format in ${jsonPath}`);
  }
  const binPath = jsonPath.replace(/\.json$/i, ".bin");
  return {
    tensors: meta.tensors.map((tensor) => ({
      name: tensor.name,
      dtype: tensor.dtype,
      shape: [...tensor.shape],
      offset: tensor.offset,
      nbytes: tensor.nbytes,
    })),
    totalBytes: meta.total_bytes,
    data: fs.readFileSync(binPath),
  };
}

async function buildFlowStateFromAudioConditioning(
  flowMain: ort.InferenceSession,
  conditioning: AudioConditioning,
  textTokenCount: number,
  maxGenLen: number,
): Promise<VoiceState> {
  const requiredCacheLength = conditioning.frames + textTokenCount + maxGenLen;
  let flowState = initFlowState(flowMain, requiredCacheLength);
  flowState = await runFlowPrompt(flowMain, flowState, conditioning.embeddings, conditioning.frames);
  return flowState;
}

function buildStateFeeds(state: VoiceState): Record<string, ort.Tensor> {
  const feeds: Record<string, ort.Tensor> = {};
  for (const [name, tensorInfo] of state.entries()) {
    if (tensorInfo.dtype === "float32") {
      feeds[name] = tensorFromFloat(tensorInfo.data as Float32Array, tensorInfo.shape);
    } else if (tensorInfo.dtype === "int64") {
      feeds[name] = tensorFromInt64(tensorInfo.data as BigInt64Array, tensorInfo.shape);
    } else {
      feeds[name] = tensorFromBool(tensorInfo.data as Uint8Array, tensorInfo.shape);
    }
  }
  return feeds;
}

function stateFromOutputs(prefix: string, outputNames: readonly string[], outputs: Record<string, ort.Tensor>): VoiceState {
  const state = new Map<string, VoiceStateTensor>();
  for (const name of outputNames.filter((item) => item.startsWith(prefix))) {
    const tensor = outputs[name];
    if (!tensor) {
      throw new Error(`Missing state output ${name}`);
    }
    const inputName = name.replace(/^out_/, "");
    if (tensor.type === "float32") {
      state.set(inputName, {
        name: inputName,
        dtype: "float32",
        shape: tensor.dims as number[],
        data: new Float32Array(tensor.data as Float32Array),
      });
    } else if (tensor.type === "int64") {
      state.set(inputName, {
        name: inputName,
        dtype: "int64",
        shape: tensor.dims as number[],
        data: new BigInt64Array(tensor.data as BigInt64Array),
      });
    } else if (tensor.type === "bool") {
      state.set(inputName, {
        name: inputName,
        dtype: "bool",
        shape: tensor.dims as number[],
        data: new Uint8Array(tensor.data as Uint8Array),
      });
    } else {
      throw new Error(`Unsupported output tensor type ${tensor.type}`);
    }
  }
  return state;
}

async function runTextConditioner(
  session: ort.InferenceSession,
  tokenIds: bigint[],
): Promise<Float32Array> {
  const ids = new BigInt64Array(tokenIds);
  const outputs = await timedRun("text_conditioner", session, {
    token_ids: tensorFromInt64(ids, [1, tokenIds.length]),
  });
  return outputs.embeddings.data as Float32Array;
}

async function runFlowPrompt(
  session: ort.InferenceSession,
  state: VoiceState,
  promptEmbeddings: Float32Array,
  promptLen: number,
): Promise<VoiceState> {
  const outputs = await timedRun("flow_lm_main", session, {
    sequence: tensorFromFloat(new Float32Array(0), [1, 0, LATENT_DIM]),
    text_embeddings: tensorFromFloat(promptEmbeddings, [1, promptLen, TEXT_DIM]),
    ...buildStateFeeds(state),
  });
  return stateFromOutputs("out_state_", session.outputNames, outputs);
}

async function runFlowStep(
  flowMain: ort.InferenceSession,
  flowNet: ort.InferenceSession,
  state: VoiceState,
  previousLatent: Float32Array,
  flowSteps: number,
  temperature: number,
): Promise<{ latent: Float32Array; isEos: boolean; nextState: VoiceState }> {
  const mainOutputs = await timedRun("flow_lm_main", flowMain, {
    sequence: tensorFromFloat(previousLatent, [1, 1, LATENT_DIM]),
    text_embeddings: tensorFromFloat(new Float32Array(0), [1, 0, TEXT_DIM]),
    ...buildStateFeeds(state),
  });

  const conditioning = mainOutputs.conditioning.data as Float32Array;
  const eosLogit = (mainOutputs.eos_logit.data as Float32Array)[0];
  const nextState = stateFromOutputs("out_state_", flowMain.outputNames, mainOutputs);

  const latent = makeNoiseTensor(Math.sqrt(Math.max(0, temperature)));
  const stepSize = 1.0 / flowSteps;
  for (let step = 0; step < flowSteps; step += 1) {
    const s = step / flowSteps;
    const t = s + stepSize;
    const flowOutputs = await timedRun("flow_lm_flow", flowNet, {
      c: tensorFromFloat(conditioning, [1, TEXT_DIM]),
      s: tensorFromFloat(new Float32Array([s]), [1, 1]),
      t: tensorFromFloat(new Float32Array([t]), [1, 1]),
      x: tensorFromFloat(latent, [1, LATENT_DIM]),
    });
    const flowDir = flowOutputs.flow_dir.data as Float32Array;
    for (let i = 0; i < LATENT_DIM; i += 1) {
      latent[i] += flowDir[i] * stepSize;
    }
  }
  return { latent, isEos: eosLogit > -4.0, nextState };
}

async function runMimiDecoder(
  session: ort.InferenceSession,
  state: VoiceState,
  latents: Float32Array,
  frameCount: number,
): Promise<{ audio: Float32Array; nextState: VoiceState }> {
  const outputs = await timedRun("mimi_decoder", session, {
    latent: tensorFromFloat(latents, [1, frameCount, LATENT_DIM]),
    ...buildStateFeeds(state),
  });
  const audio = outputs.audio_frame.data as Float32Array;
  const nextState = stateFromOutputs("out_state_", session.outputNames, outputs);
  return { audio, nextState };
}

function concatAudio(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const output = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

function writeWav(filePath: string, audio: Float32Array, sampleRate: number): void {
  const buffer = Buffer.alloc(44 + audio.length * 2);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + audio.length * 2, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(audio.length * 2, 40);
  for (let i = 0; i < audio.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, audio[i]));
    buffer.writeInt16LE(Math.round(sample * 32767), 44 + i * 2);
  }
  fs.writeFileSync(filePath, buffer);
}

function writeFloat32Wav(filePath: string, audio: Float32Array, sampleRate: number): void {
  const buffer = Buffer.alloc(44 + audio.length * 4);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + audio.length * 4, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(3, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 4, 28);
  buffer.writeUInt16LE(4, 32);
  buffer.writeUInt16LE(32, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(audio.length * 4, 40);
  for (let i = 0; i < audio.length; i += 1) {
    buffer.writeFloatLE(audio[i], 44 + i * 4);
  }
  fs.writeFileSync(filePath, buffer);
}

function writeRawFloat32(pathStem: string, audio: Float32Array): string {
  fs.writeFileSync(`${pathStem}.bin`, Buffer.from(audio.buffer, audio.byteOffset, audio.byteLength));
  fs.writeFileSync(
    `${pathStem}.json`,
    JSON.stringify({ shape: [audio.length], dtype: "float32" }, null, 2) + "\n",
    "utf-8",
  );
  return `${pathStem}.bin`;
}

function audioStats(audio: Float32Array): { min: number; max: number; rms: number } {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sumSquares = 0;
  for (let i = 0; i < audio.length; i += 1) {
    const sample = audio[i];
    if (sample < min) {
      min = sample;
    }
    if (sample > max) {
      max = sample;
    }
    sumSquares += sample * sample;
  }
  return {
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
    rms: audio.length === 0 ? 0 : Math.sqrt(sumSquares / audio.length),
  };
}

function readWav(filePath: string): WavData {
  const buffer = fs.readFileSync(filePath);
  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error(`Unsupported WAV file: ${filePath}`);
  }
  const audioFormat = buffer.readUInt16LE(20);
  const channels = buffer.readUInt16LE(22);
  const sampleRate = buffer.readUInt32LE(24);
  const bitsPerSample = buffer.readUInt16LE(34);
  const dataSize = buffer.readUInt32LE(40);
  const dataStart = 44;
  const frameCount = dataSize / (bitsPerSample / 8) / channels;
  const mono = new Float32Array(frameCount);

  if (audioFormat === 1 && bitsPerSample === 16) {
    for (let frame = 0; frame < frameCount; frame += 1) {
      let sum = 0;
      for (let channel = 0; channel < channels; channel += 1) {
        const offset = dataStart + (frame * channels + channel) * 2;
        sum += buffer.readInt16LE(offset) / 32768;
      }
      mono[frame] = sum / channels;
    }
  } else if (audioFormat === 3 && bitsPerSample === 32) {
    for (let frame = 0; frame < frameCount; frame += 1) {
      let sum = 0;
      for (let channel = 0; channel < channels; channel += 1) {
        const offset = dataStart + (frame * channels + channel) * 4;
        sum += buffer.readFloatLE(offset);
      }
      mono[frame] = sum / channels;
    }
  } else {
    throw new Error(`Unsupported WAV encoding format=${audioFormat} bits=${bitsPerSample}`);
  }

  return { sampleRate, channels, samples: mono };
}

function resampleLinear(input: Float32Array, sourceRate: number, targetRate: number): Float32Array {
  if (sourceRate === targetRate) {
    return input;
  }
  const ratio = targetRate / sourceRate;
  const outputLength = Math.max(1, Math.round(input.length * ratio));
  const output = new Float32Array(outputLength);
  for (let index = 0; index < outputLength; index += 1) {
    const position = index / ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, input.length - 1);
    const frac = position - left;
    output[index] = input[left] * (1 - frac) + input[right] * frac;
  }
  return output;
}

async function runScenario(
  scenarioName: string,
  flowMain: ort.InferenceSession,
  flowNet: ort.InferenceSession,
  textConditioner: ort.InferenceSession,
  mimiDecoder: ort.InferenceSession,
  tokenizer: Tokenizer,
  text: string,
  conditioning: ScenarioConditioning,
  outputPath: string,
  flowSteps: number,
  temperature: number,
): Promise<ScenarioStats> {
  const scenarioStart = performance.now();
  const beforeCalls = snapshotModelStats();
  const { framesAfterEos } = prepareTextPrompt(text);
  const encoding = tokenizer.encode(text, { add_special_tokens: false });
  const tokenIds = encoding.ids.map((id: number) => BigInt(id));
  const decoderMaxLatents = Math.floor(metadataShape(mimiDecoder, "state_19")[2] / MIMI_STEPS_PER_LATENT);
  const maxGenLen = Math.min(estimateMaxGenLen(tokenIds.length), decoderMaxLatents);

  let flowState: VoiceState;
  if (conditioning.value.kind === "serialized") {
    const bundle = loadVoiceStateBundleV2(conditioning.value.jsonPath);
    const currentEnd = readCurrentEndFromBundle(bundle);
    const requiredCacheLength = currentEnd + tokenIds.length + maxGenLen;
    flowState = buildSeededFlowState(flowMain, bundle, requiredCacheLength);
    if (DEBUG_COMPARE) {
      console.log(`${scenarioName}: using serialized flow state bundle currentEnd=${currentEnd} cacheLength=${requiredCacheLength}`);
    }
  } else {
    const currentEnd = readCurrentEndFromBundle(conditioning.value.bundle);
    const requiredCacheLength = currentEnd + tokenIds.length + maxGenLen;
    flowState = buildSeededFlowState(flowMain, conditioning.value.bundle, requiredCacheLength);
    if (DEBUG_COMPARE) {
      console.log(`${scenarioName}: using in-memory flow state bundle currentEnd=${currentEnd} cacheLength=${requiredCacheLength}`);
    }
  }
  const textEmbeddings = await runTextConditioner(textConditioner, tokenIds);
  if (DEBUG_COMPARE) {
    console.log(`${scenarioName}: text token count=${tokenIds.length}`);
  }
  flowState = await runFlowPrompt(flowMain, flowState, textEmbeddings, tokenIds.length);

  let previousLatent = new Float32Array(LATENT_DIM);
  previousLatent.fill(Number.NaN);
  let eosStep: number | null = null;
  let mimiState = initMimiState(mimiDecoder);
  const audioChunks: Float32Array[] = [];

  for (let generationStep = 0; generationStep < maxGenLen; generationStep += 1) {
    const { latent, isEos, nextState } = await runFlowStep(flowMain, flowNet, flowState, previousLatent, flowSteps, temperature);
    if (DEBUG_COMPARE) {
      const finite = latent.every((value) => Number.isFinite(value));
      console.log(`${scenarioName}: step=${generationStep} eos=${isEos} latentFinite=${finite}`);
    }
    flowState = nextState;
    if (isEos && eosStep === null) {
      eosStep = generationStep;
    }
    if (eosStep !== null && generationStep >= eosStep + framesAfterEos) {
      break;
    }
    const decoded = await runMimiDecoder(mimiDecoder, mimiState, latent, 1);
    mimiState = decoded.nextState;
    audioChunks.push(new Float32Array(decoded.audio));
    previousLatent = new Float32Array(latent);
  }

  const audio = concatAudio(audioChunks);
  const rawPath = writeRawFloat32(outputPath.replace(/\.wav$/i, "_raw"), audio);
  writeWav(outputPath, audio, SAMPLE_RATE);
  const outputPathFloat32 = outputPath.replace(/\.wav$/i, "_f32.wav");
  writeFloat32Wav(outputPathFloat32, audio, SAMPLE_RATE);

  const afterCalls = snapshotModelStats();
  const scenarioModels: ScenarioStats["models"] = {};
  for (const modelName of Object.keys(modelStats)) {
    const deltaCalls = afterCalls[modelName].calls - beforeCalls[modelName].calls;
    const deltaMs = afterCalls[modelName].totalInferenceMs - beforeCalls[modelName].totalInferenceMs;
    scenarioModels[modelName] = {
      calls: deltaCalls,
      totalInferenceMs: deltaMs,
      averageInferenceMs: deltaCalls === 0 ? 0 : deltaMs / deltaCalls,
    };
  }

  const totalMs = performance.now() - scenarioStart;
  console.log(`${scenarioName}: wrote ${outputPath}`);
  return {
    totalMs,
    audioSeconds: audio.length / SAMPLE_RATE,
    models: scenarioModels,
    outputPath,
    outputPathFloat32,
    rawPath,
    audioStats: audioStats(audio),
  };
}

function snapshotModelStats(): Record<string, { calls: number; totalInferenceMs: number }> {
  const snapshot: Record<string, { calls: number; totalInferenceMs: number }> = {};
  for (const [modelName, stats] of Object.entries(modelStats)) {
    snapshot[modelName] = { calls: stats.calls, totalInferenceMs: stats.totalInferenceMs };
  }
  return snapshot;
}

function summarizeMountStats(): Record<string, { mountMs: number; totalInferenceMs: number; calls: number; averageInferenceMs: number }> {
  const summary: Record<string, { mountMs: number; totalInferenceMs: number; calls: number; averageInferenceMs: number }> = {};
  for (const [modelName, stats] of Object.entries(modelStats)) {
    summary[modelName] = {
      mountMs: stats.mountMs,
      totalInferenceMs: stats.totalInferenceMs,
      calls: stats.calls,
      averageInferenceMs: stats.calls === 0 ? 0 : stats.totalInferenceMs / stats.calls,
    };
  }
  return summary;
}

async function buildCloneState(
  mimiEncoder: ort.InferenceSession,
  clonePromptPath: string,
): Promise<AudioConditioning> {
  const wav = readWav(clonePromptPath);
  const mono24k = resampleLinear(wav.samples, wav.sampleRate, SAMPLE_RATE);
  const encoderOutputs = await timedRun("mimi_encoder", mimiEncoder, {
    audio: tensorFromFloat(mono24k, [1, 1, mono24k.length]),
  });
  const latents = encoderOutputs.latents.data as Float32Array;
  const promptLen = (encoderOutputs.latents.dims as number[])[1];
  return { embeddings: new Float32Array(latents), frames: promptLen };
}

async function main(): Promise<void> {
  ensureDir(OUTPUT_DIR);

  const precisionArg = (process.argv.find((arg) => arg.startsWith("--precision="))?.split("=")[1] ?? DEFAULT_PRECISION) as Precision;
  const builtinVoice = process.argv.find((arg) => arg.startsWith("--builtin-voice="))?.split("=")[1] ?? DEFAULT_BUILTIN_VOICE;
  const comparisonText = process.argv.find((arg) => arg.startsWith("--text="))?.split("=")[1] ?? DEFAULT_TEXT;
  const flowSteps = Number.parseInt(process.argv.find((arg) => arg.startsWith("--flow-steps="))?.split("=")[1] ?? `${DEFAULT_FLOW_STEPS}`, 10);
  const temperature = Number.parseFloat(process.argv.find((arg) => arg.startsWith("--temperature="))?.split("=")[1] ?? `${DEFAULT_TEMPERATURE}`);

  if (!Number.isFinite(flowSteps) || flowSteps < 1) {
    throw new Error(`Invalid --flow-steps value: ${flowSteps}`);
  }
  if (!Number.isFinite(temperature) || temperature < 0) {
    throw new Error(`Invalid --temperature value: ${temperature}`);
  }

  if (!fs.existsSync(CLONE_PROMPT_PATH)) {
    throw new Error(`Missing clone prompt at ${CLONE_PROMPT_PATH}. Run the Python benchmark setup first.`);
  }

  const tokenizerJson = readJson<object>(path.join(HF_DIR, "tokenizer.json"));
  const tokenizerConfig = readJson<object>(path.join(HF_DIR, "tokenizer_config.json"));
  const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
  const textConditioner = await mountSession("text_conditioner", modelPath("text_conditioner", precisionArg));
  const flowMain = await mountSession("flow_lm_main", modelPath("flow_lm_main", precisionArg));
  const flowNet = await mountSession("flow_lm_flow", modelPath("flow_lm_flow", precisionArg));
  const mimiDecoder = await mountSession("mimi_decoder", modelPath("mimi_decoder", precisionArg));
  const mimiEncoder = await mountSession("mimi_encoder", modelPath("mimi_encoder", precisionArg));

  const builtinStateJson = path.join(HF_DIR, "embeddings_v2", `${builtinVoice}.json`);
  if (!fs.existsSync(builtinStateJson)) {
    throw new Error(`Missing v2 built-in voice state for ${builtinVoice}: ${builtinStateJson}`);
  }
  const builtinConditioning: ScenarioConditioning = {
    kind: "flow_state",
    value: { kind: "serialized", jsonPath: builtinStateJson },
  };

  const builtinScenario = await runScenario(
    "node_builtin",
    flowMain,
    flowNet,
    textConditioner,
    mimiDecoder,
    tokenizer,
    comparisonText,
    builtinConditioning,
    BUILTIN_OUTPUT_PATH,
    flowSteps,
    temperature,
  );

  const cloneAudioConditioning = await buildCloneState(mimiEncoder, CLONE_PROMPT_PATH);
  const cloneTokenIds = tokenizer.encode(comparisonText, { add_special_tokens: false }).ids;
  const cloneDecoderMaxLatents = Math.floor(metadataShape(mimiDecoder, "state_19")[2] / MIMI_STEPS_PER_LATENT);
  const cloneMaxGenLen = Math.min(estimateMaxGenLen(cloneTokenIds.length), cloneDecoderMaxLatents);
  const cloneFlowState = await buildFlowStateFromAudioConditioning(
    flowMain,
    cloneAudioConditioning,
    cloneTokenIds.length,
    cloneMaxGenLen,
  );
  const cloneFlowStateBundle = buildVoiceStateBundleFromState(cloneFlowState);
  const cloneConditioning: ScenarioConditioning = {
    kind: "flow_state",
    value: { kind: "runtime", bundle: cloneFlowStateBundle },
  };
  const cloneScenario = await runScenario(
    "node_voice_clone",
    flowMain,
    flowNet,
    textConditioner,
    mimiDecoder,
    tokenizer,
    comparisonText,
    cloneConditioning,
    CLONE_OUTPUT_PATH,
    flowSteps,
    temperature,
  );

  const report = {
    precision: precisionArg,
    flowSteps,
    temperature,
    builtinConditioningMode: "state",
    builtinVoice,
    comparisonText,
    clonePromptPath: CLONE_PROMPT_PATH,
    modelStats: summarizeMountStats(),
    scenarios: {
      builtin: builtinScenario,
      voiceClone: cloneScenario,
    },
  };

  fs.writeFileSync(REPORT_PATH, JSON.stringify(report, null, 2) + "\n", "utf-8");
  console.log(`Wrote ${REPORT_PATH}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});