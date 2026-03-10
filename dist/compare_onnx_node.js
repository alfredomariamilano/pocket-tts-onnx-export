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
const CLONE_PROMPT_PATH = path.join(OUTPUT_DIR, "clone_prompt.wav");
const BUILTIN_OUTPUT_PATH = path.join(OUTPUT_DIR, "node_builtin.wav");
const CLONE_OUTPUT_PATH = path.join(OUTPUT_DIR, "node_voice_clone.wav");
const REPORT_PATH = path.join(OUTPUT_DIR, "node_onnx_report.json");
const DEFAULT_TEXT = "This is a direct quality comparison between the ONNX Node runtime and the native Python Pocket TTS runtime.";
const DEBUG_COMPARE = process.env.DEBUG_COMPARE === "1";
const FLOW_STEPS = 1;
const MIMI_STEPS_PER_LATENT = 16;
const modelStats = {};
function ensureDir(dirPath) {
    fs.mkdirSync(dirPath, { recursive: true });
}
function suffixForPrecision(precision) {
    if (precision === "fp32") {
        return ".onnx";
    }
    return `_${precision}.onnx`;
}
function modelPath(baseName, precision) {
    return path.join(ONNX_DIR, `${baseName}${suffixForPrecision(precision)}`);
}
async function mountSession(modelName, filePath) {
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
async function timedRun(modelName, session, feeds) {
    const start = performance.now();
    const result = await session.run(feeds);
    const elapsed = performance.now() - start;
    modelStats[modelName].calls += 1;
    modelStats[modelName].totalInferenceMs += elapsed;
    return result;
}
function normalRandom() {
    let u = 0;
    let v = 0;
    while (u === 0)
        u = Math.random();
    while (v === 0)
        v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function makeNoiseTensor(std) {
    const values = new Float32Array(LATENT_DIM);
    for (let i = 0; i < values.length; i += 1) {
        values[i] = normalRandom() * std;
    }
    return values;
}
function prepareTextPrompt(text) {
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
function estimateMaxGenLen(tokenCount) {
    const genSeconds = tokenCount / 3.0 + 2.0;
    return Math.ceil(genSeconds * 12.5);
}
function estimateMinGenerationSteps(tokenCount, decoderMaxLatents, framesAfterEos) {
    const heuristic = Math.max(8, Math.ceil(tokenCount * 1.5));
    return Math.min(heuristic, Math.max(1, decoderMaxLatents - framesAfterEos - 1));
}
function tensorFromFloat(data, dims) {
    return new ort.Tensor("float32", data, dims);
}
function tensorFromInt64(data, dims) {
    return new ort.Tensor("int64", data, dims);
}
function tensorFromBool(data, dims) {
    return new ort.Tensor("bool", data, dims);
}
function createFloatState(dims, fillNaN) {
    const total = dims.reduce((acc, value) => acc * value, 1);
    const values = new Float32Array(total);
    if (fillNaN) {
        values.fill(Number.NaN);
    }
    return values;
}
function getInputMeta(session, inputName) {
    const rawMetadata = session.inputMetadata;
    if (Array.isArray(rawMetadata)) {
        return rawMetadata.find((entry) => entry.name === inputName);
    }
    return rawMetadata[inputName];
}
function normalizeTensorType(type) {
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
function metadataShape(session, inputName) {
    const meta = getInputMeta(session, inputName);
    const dims = meta?.dimensions ?? meta?.shape;
    if (!dims) {
        throw new Error(`Missing input metadata for ${inputName}`);
    }
    return dims.map((dim) => (typeof dim === "number" ? dim : 0));
}
function initState(session) {
    const state = new Map();
    for (const inputName of session.inputNames.filter((name) => name.startsWith("state_"))) {
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
        }
        else if (tensorType === "int64") {
            state.set(inputName, {
                name: inputName,
                dtype: "int64",
                shape,
                data: new BigInt64Array(shape.reduce((acc, value) => acc * value, 1)),
            });
        }
        else if (tensorType === "bool") {
            state.set(inputName, {
                name: inputName,
                dtype: "bool",
                shape,
                data: new Uint8Array(shape.reduce((acc, value) => acc * value, 1)),
            });
        }
        else {
            throw new Error(`Unsupported state type ${meta.type} for ${inputName}`);
        }
    }
    return state;
}
function initFlowState(flowMainSession, cacheLength) {
    const state = new Map();
    for (const inputName of flowMainSession.inputNames.filter((name) => name.startsWith("state_"))) {
        const meta = getInputMeta(flowMainSession, inputName);
        if (!meta) {
            throw new Error(`Missing input metadata for ${inputName}`);
        }
        const shape = metadataShape(flowMainSession, inputName);
        const tensorType = normalizeTensorType(meta.type);
        if (tensorType === "float32") {
            const targetShape = shape.length === 5 ? [shape[0], shape[1], cacheLength, shape[3], shape[4]] : shape;
            state.set(inputName, {
                name: inputName,
                dtype: "float32",
                shape: targetShape,
                data: createFloatState(targetShape, true),
            });
        }
        else if (tensorType === "int64") {
            state.set(inputName, {
                name: inputName,
                dtype: "int64",
                shape,
                data: new BigInt64Array(shape.reduce((acc, value) => acc * value, 1)),
            });
        }
        else if (tensorType === "bool") {
            state.set(inputName, {
                name: inputName,
                dtype: "bool",
                shape,
                data: new Uint8Array(shape.reduce((acc, value) => acc * value, 1)),
            });
        }
        else {
            throw new Error(`Unsupported state type ${meta.type} for ${inputName}`);
        }
    }
    return state;
}
function initMimiState(mimiDecoderSession) {
    return initState(mimiDecoderSession);
}
function readJson(filePath) {
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}
function loadAudioConditioning(basePath) {
    const meta = readJson(`${basePath}.json`);
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
function readTypedSlice(blob, offset, nbytes, dtype) {
    const slice = blob.subarray(offset, offset + nbytes);
    if (dtype === "float32") {
        return new Float32Array(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength));
    }
    if (dtype === "int64") {
        return new BigInt64Array(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength));
    }
    return new Uint8Array(slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength));
}
function expandFloatCacheToModelShape(compact, compactShape, targetShape) {
    const expanded = createFloatState(targetShape, true);
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
function readFlowStateCurrentEnd(jsonPath) {
    const meta = readJson(jsonPath);
    const currentEndTensor = meta.tensors.find((entry) => entry.source_key === "current_end")
        ?? meta.tensors.find((entry) => entry.name === "state_1");
    if (!currentEndTensor || currentEndTensor.dtype !== "int64" || currentEndTensor.shape.reduce((acc, value) => acc * value, 1) !== 1) {
        throw new Error(`Could not locate current_end tensor in ${jsonPath}`);
    }
    const binPath = jsonPath.replace(/\.json$/i, ".bin");
    const blob = fs.readFileSync(binPath);
    const value = readTypedSlice(blob, currentEndTensor.offset, currentEndTensor.nbytes, "int64");
    return Number(value[0]);
}
function loadFlowStateV2(jsonPath, flowMainSession, targetCacheLength) {
    const meta = readJson(jsonPath);
    if (meta.format !== "pocket_tts_flow_state_v2") {
        throw new Error(`Unsupported flow state format in ${jsonPath}`);
    }
    const binPath = jsonPath.replace(/\.json$/i, ".bin");
    const blob = fs.readFileSync(binPath);
    const state = new Map();
    for (const entry of meta.tensors) {
        const modelShape = metadataShape(flowMainSession, entry.name);
        let data = readTypedSlice(blob, entry.offset, entry.nbytes, entry.dtype);
        let shape = [...entry.shape];
        if (entry.dtype === "float32" && shape.length === 5) {
            const targetShape = [shape[0], shape[1], targetCacheLength, shape[3], shape[4]];
            if (targetCacheLength < shape[2]) {
                throw new Error(`Target FlowLM cache length ${targetCacheLength} is smaller than serialized cache length ${shape[2]} for ${entry.name}`);
            }
            if (shape[2] !== targetShape[2]) {
                data = expandFloatCacheToModelShape(data, shape, targetShape);
                shape = targetShape;
            }
        }
        else if (entry.dtype === "float32" && shape.length === 5 && modelShape[2] > 0 && shape[2] !== modelShape[2]) {
            data = expandFloatCacheToModelShape(data, shape, modelShape);
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
function cloneState(state) {
    const cloned = new Map();
    for (const [name, tensorInfo] of state.entries()) {
        if (tensorInfo.dtype === "float32") {
            cloned.set(name, { ...tensorInfo, data: new Float32Array(tensorInfo.data) });
        }
        else if (tensorInfo.dtype === "int64") {
            cloned.set(name, { ...tensorInfo, data: new BigInt64Array(tensorInfo.data) });
        }
        else {
            cloned.set(name, { ...tensorInfo, data: new Uint8Array(tensorInfo.data) });
        }
    }
    return cloned;
}
function buildStateFeeds(state) {
    const feeds = {};
    for (const [name, tensorInfo] of state.entries()) {
        if (tensorInfo.dtype === "float32") {
            feeds[name] = tensorFromFloat(tensorInfo.data, tensorInfo.shape);
        }
        else if (tensorInfo.dtype === "int64") {
            feeds[name] = tensorFromInt64(tensorInfo.data, tensorInfo.shape);
        }
        else {
            feeds[name] = tensorFromBool(tensorInfo.data, tensorInfo.shape);
        }
    }
    return feeds;
}
function stateFromOutputs(prefix, outputNames, outputs) {
    const state = new Map();
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
                shape: tensor.dims,
                data: new Float32Array(tensor.data),
            });
        }
        else if (tensor.type === "int64") {
            state.set(inputName, {
                name: inputName,
                dtype: "int64",
                shape: tensor.dims,
                data: new BigInt64Array(tensor.data),
            });
        }
        else if (tensor.type === "bool") {
            state.set(inputName, {
                name: inputName,
                dtype: "bool",
                shape: tensor.dims,
                data: new Uint8Array(tensor.data),
            });
        }
        else {
            throw new Error(`Unsupported output tensor type ${tensor.type}`);
        }
    }
    return state;
}
async function runTextConditioner(session, tokenIds) {
    const ids = new BigInt64Array(tokenIds);
    const outputs = await timedRun("text_conditioner", session, {
        token_ids: tensorFromInt64(ids, [1, tokenIds.length]),
    });
    return outputs.embeddings.data;
}
async function runFlowPrompt(session, state, promptEmbeddings, promptLen) {
    const outputs = await timedRun("flow_lm_main", session, {
        sequence: tensorFromFloat(new Float32Array(0), [1, 0, LATENT_DIM]),
        text_embeddings: tensorFromFloat(promptEmbeddings, [1, promptLen, TEXT_DIM]),
        ...buildStateFeeds(state),
    });
    return stateFromOutputs("out_state_", session.outputNames, outputs);
}
async function runFlowStep(flowMain, flowNet, state, previousLatent) {
    const mainOutputs = await timedRun("flow_lm_main", flowMain, {
        sequence: tensorFromFloat(previousLatent, [1, 1, LATENT_DIM]),
        text_embeddings: tensorFromFloat(new Float32Array(0), [1, 0, TEXT_DIM]),
        ...buildStateFeeds(state),
    });
    const conditioning = mainOutputs.conditioning.data;
    const eosLogit = mainOutputs.eos_logit.data[0];
    const nextState = stateFromOutputs("out_state_", flowMain.outputNames, mainOutputs);
    const latent = makeNoiseTensor(Math.sqrt(0.7));
    const stepSize = 1.0 / FLOW_STEPS;
    for (let step = 0; step < FLOW_STEPS; step += 1) {
        const s = step / FLOW_STEPS;
        const t = s + stepSize;
        const flowOutputs = await timedRun("flow_lm_flow", flowNet, {
            c: tensorFromFloat(conditioning, [1, TEXT_DIM]),
            s: tensorFromFloat(new Float32Array([s]), [1, 1]),
            t: tensorFromFloat(new Float32Array([t]), [1, 1]),
            x: tensorFromFloat(latent, [1, LATENT_DIM]),
        });
        const flowDir = flowOutputs.flow_dir.data;
        for (let i = 0; i < LATENT_DIM; i += 1) {
            latent[i] += flowDir[i] * stepSize;
        }
    }
    return { latent, isEos: eosLogit > -4.0, nextState };
}
async function runMimiDecoder(session, state, latent) {
    const outputs = await timedRun("mimi_decoder", session, {
        latent: tensorFromFloat(latent, [1, 1, LATENT_DIM]),
        ...buildStateFeeds(state),
    });
    const audio = outputs.audio_frame.data;
    const nextState = stateFromOutputs("out_state_", session.outputNames, outputs);
    return { audio, nextState };
}
function concatAudio(chunks) {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const output = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        output.set(chunk, offset);
        offset += chunk.length;
    }
    return output;
}
function writeWav(filePath, audio, sampleRate) {
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
function readWav(filePath) {
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
    }
    else if (audioFormat === 3 && bitsPerSample === 32) {
        for (let frame = 0; frame < frameCount; frame += 1) {
            let sum = 0;
            for (let channel = 0; channel < channels; channel += 1) {
                const offset = dataStart + (frame * channels + channel) * 4;
                sum += buffer.readFloatLE(offset);
            }
            mono[frame] = sum / channels;
        }
    }
    else {
        throw new Error(`Unsupported WAV encoding format=${audioFormat} bits=${bitsPerSample}`);
    }
    return { sampleRate, channels, samples: mono };
}
function resampleLinear(input, sourceRate, targetRate) {
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
async function runScenario(scenarioName, flowMain, flowNet, textConditioner, mimiDecoder, tokenizer, text, conditioning, outputPath) {
    const scenarioStart = performance.now();
    const beforeCalls = snapshotModelStats();
    const { prepared, framesAfterEos } = prepareTextPrompt(text);
    const encoding = tokenizer.encode(prepared, { add_special_tokens: false });
    const tokenIds = encoding.ids.map((id) => BigInt(id));
    const decoderMaxLatents = Math.floor(metadataShape(mimiDecoder, "state_19")[2] / MIMI_STEPS_PER_LATENT);
    const maxGenLen = Math.min(estimateMaxGenLen(tokenIds.length), decoderMaxLatents);
    const minStepsBeforeEos = estimateMinGenerationSteps(tokenIds.length, decoderMaxLatents, framesAfterEos);
    let flowState;
    if (conditioning.kind === "flow_state") {
        const currentEnd = readFlowStateCurrentEnd(conditioning.value.jsonPath);
        const requiredCacheLength = currentEnd + tokenIds.length + maxGenLen;
        flowState = loadFlowStateV2(conditioning.value.jsonPath, flowMain, requiredCacheLength);
        if (DEBUG_COMPARE) {
            console.log(`${scenarioName}: using precomputed flow state currentEnd=${currentEnd} cacheLength=${requiredCacheLength}`);
        }
    }
    else {
        const requiredCacheLength = conditioning.value.frames + tokenIds.length + maxGenLen;
        flowState = initFlowState(flowMain, requiredCacheLength);
        if (DEBUG_COMPARE) {
            console.log(`${scenarioName}: audio conditioning frames=${conditioning.value.frames} cacheLength=${requiredCacheLength}`);
        }
        flowState = await runFlowPrompt(flowMain, flowState, conditioning.value.embeddings, conditioning.value.frames);
    }
    const textEmbeddings = await runTextConditioner(textConditioner, tokenIds);
    if (DEBUG_COMPARE) {
        console.log(`${scenarioName}: text token count=${tokenIds.length}`);
    }
    flowState = await runFlowPrompt(flowMain, flowState, textEmbeddings, tokenIds.length);
    let mimiState = initMimiState(mimiDecoder);
    let previousLatent = new Float32Array(LATENT_DIM);
    previousLatent.fill(Number.NaN);
    let eosStep = null;
    const audioChunks = [];
    for (let generationStep = 0; generationStep < maxGenLen; generationStep += 1) {
        const { latent, isEos, nextState } = await runFlowStep(flowMain, flowNet, flowState, previousLatent);
        if (DEBUG_COMPARE) {
            const finite = latent.every((value) => Number.isFinite(value));
            console.log(`${scenarioName}: step=${generationStep} eos=${isEos} latentFinite=${finite} minStepsBeforeEos=${minStepsBeforeEos}`);
        }
        flowState = nextState;
        if (generationStep >= minStepsBeforeEos && isEos && eosStep === null) {
            eosStep = generationStep;
        }
        if (eosStep !== null && generationStep >= eosStep + framesAfterEos) {
            break;
        }
        const decoded = await runMimiDecoder(mimiDecoder, mimiState, latent);
        mimiState = decoded.nextState;
        audioChunks.push(new Float32Array(decoded.audio));
        previousLatent = new Float32Array(latent);
    }
    const audio = concatAudio(audioChunks);
    writeWav(outputPath, audio, SAMPLE_RATE);
    const afterCalls = snapshotModelStats();
    const scenarioModels = {};
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
    };
}
function snapshotModelStats() {
    const snapshot = {};
    for (const [modelName, stats] of Object.entries(modelStats)) {
        snapshot[modelName] = { calls: stats.calls, totalInferenceMs: stats.totalInferenceMs };
    }
    return snapshot;
}
function summarizeMountStats() {
    const summary = {};
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
async function buildCloneState(mimiEncoder, clonePromptPath) {
    const wav = readWav(clonePromptPath);
    const mono24k = resampleLinear(wav.samples, wav.sampleRate, SAMPLE_RATE);
    const encoderOutputs = await timedRun("mimi_encoder", mimiEncoder, {
        audio: tensorFromFloat(mono24k, [1, 1, mono24k.length]),
    });
    const latents = encoderOutputs.latents.data;
    const promptLen = encoderOutputs.latents.dims[1];
    return { embeddings: new Float32Array(latents), frames: promptLen };
}
async function main() {
    ensureDir(OUTPUT_DIR);
    const precisionArg = (process.argv.find((arg) => arg.startsWith("--precision="))?.split("=")[1] ?? DEFAULT_PRECISION);
    const builtinVoice = process.argv.find((arg) => arg.startsWith("--builtin-voice="))?.split("=")[1] ?? DEFAULT_BUILTIN_VOICE;
    const comparisonText = process.argv.find((arg) => arg.startsWith("--text="))?.split("=")[1] ?? DEFAULT_TEXT;
    if (!fs.existsSync(CLONE_PROMPT_PATH)) {
        throw new Error(`Missing clone prompt at ${CLONE_PROMPT_PATH}. Run the Python benchmark setup first.`);
    }
    const tokenizerJson = readJson(path.join(HF_DIR, "tokenizer.json"));
    const tokenizerConfig = readJson(path.join(HF_DIR, "tokenizer_config.json"));
    const tokenizer = new Tokenizer(tokenizerJson, tokenizerConfig);
    const textConditioner = await mountSession("text_conditioner", modelPath("text_conditioner", precisionArg));
    const flowMain = await mountSession("flow_lm_main", modelPath("flow_lm_main", precisionArg));
    const flowNet = await mountSession("flow_lm_flow", modelPath("flow_lm_flow", precisionArg));
    const mimiDecoder = await mountSession("mimi_decoder", modelPath("mimi_decoder", precisionArg));
    const mimiEncoder = await mountSession("mimi_encoder", modelPath("mimi_encoder", precisionArg));
    const builtinStateJson = path.join(HF_DIR, "embeddings_v2", `${builtinVoice}.json`);
    const builtinConditioning = fs.existsSync(builtinStateJson)
        ? { kind: "flow_state", value: { jsonPath: builtinStateJson } }
        : { kind: "audio_embeddings", value: loadAudioConditioning(path.join(HF_DIR, "embeddings", builtinVoice)) };
    const builtinScenario = await runScenario("node_builtin", flowMain, flowNet, textConditioner, mimiDecoder, tokenizer, comparisonText, builtinConditioning, BUILTIN_OUTPUT_PATH);
    const cloneConditioning = {
        kind: "audio_embeddings",
        value: await buildCloneState(mimiEncoder, CLONE_PROMPT_PATH),
    };
    const cloneScenario = await runScenario("node_voice_clone", flowMain, flowNet, textConditioner, mimiDecoder, tokenizer, comparisonText, cloneConditioning, CLONE_OUTPUT_PATH);
    const report = {
        precision: precisionArg,
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
