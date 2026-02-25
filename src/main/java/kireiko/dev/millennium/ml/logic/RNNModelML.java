package kireiko.dev.millennium.ml.logic;

import kireiko.dev.millennium.ml.data.ObjectML;
import kireiko.dev.millennium.ml.data.ResultML;
import kireiko.dev.millennium.ml.logic.rnn.*;
import kireiko.dev.millennium.ml.logic.rnn.data.SequenceData;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.HybridSequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.RawSequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.SequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.data.preprocessing.StatisticalSequencePreprocessor;
import kireiko.dev.millennium.ml.logic.rnn.layers.BinaryHead;
import kireiko.dev.millennium.ml.logic.rnn.layers.LSTMLayer;
import kireiko.dev.millennium.ml.logic.rnn.layers.StackedBiLSTM;
import kireiko.dev.millennium.ml.logic.rnn.optim.AdamW;
import kireiko.dev.millennium.ml.logic.rnn.pooling.*;
import kireiko.dev.millennium.ml.logic.rnn.util.ModelIO;
import kireiko.dev.millennium.vectors.Pair;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class RNNModelML implements Millennium {

    public enum InputMode { RAW_SEQUENCE, STATISTICAL_FEATURES, HYBRID }
    public enum PoolingMode { LAST_HIDDEN, MEAN_POOLING, MAX_POOLING, ATTENTION }

    private static final int MAGIC = 0x524E4E36;
    private static final int VERSION = 7;
    private static final int LEGACY_VERSION = 6;
    private static final double MIN_LEARNING_RATE = 1e-8;
    private static final double MAX_LEARNING_RATE = 1.0;
    private static final double MAX_DROPOUT = 0.95;
    private static final double MAX_WEIGHT_DECAY = 1.0;
    private static final double MAX_GRAD_CLIP = 1_000_000.0;
    private static final double MAX_LABEL_SMOOTHING = 0.5;
    private static final double DEFAULT_LEARNING_RATE = 0.0003;
    private static final double DEFAULT_DROPOUT = 0.1;
    private static final double DEFAULT_RECURRENT_DROPOUT = 0.2;
    private static final double DEFAULT_WEIGHT_DECAY = 1e-3;
    private static final double DEFAULT_GRAD_CLIP = 5.0;
    private static final double DEFAULT_LABEL_SMOOTHING = 0.1;
    private static final int DEFAULT_INPUT_SIZE = 16;
    private static final int DEFAULT_HIDDEN_SIZE = 64;
    private static final int DEFAULT_NUM_LAYERS = 2;
    private static final int MIN_INPUT_SIZE = 2;
    private static final int MAX_INPUT_SIZE = 1024;
    private static final int MIN_HIDDEN_SIZE = 4;
    private static final int MAX_HIDDEN_SIZE = 1024;
    private static final int MIN_NUM_LAYERS = 1;
    private static final int MAX_NUM_LAYERS = 8;
    private static final int DEFAULT_CHUNK_SIZE = 150;

    private final RNNConfig cfg;
    private final Random rng;
    private final ReadWriteLock modelLock;
    private final Lock readLock;
    private final Lock writeLock;

    private final RawSequencePreprocessor rawPre;
    private final StatisticalSequencePreprocessor statPre;
    private final SequencePreprocessor hybridPre;
    private SequencePreprocessor activePre;

    private final StackedBiLSTM encoder;

    private final AttentionPooling attnPooling;
    private final PoolingStrategy lastPooling;
    private final PoolingStrategy meanPooling;
    private final PoolingStrategy maxPooling;
    private PoolingStrategy activePooling;

    private final BinaryHead head;
    private final AdamW opt;

    private long step;
    private int batchSize = 16;

    private final double[] dHeadV;
    private final double[] mHeadV;
    private final double[] vHeadV;
    private final AdamW.DoubleRef headBias;
    private final AdamW.DoubleRef mHeadBias;
    private final AdamW.DoubleRef vHeadBias;

    private final double[] dAttnW;
    private final double[] mAttnW;
    private final double[] vAttnW;
    private final AdamW.DoubleRef attnB;
    private final AdamW.DoubleRef mAttnB;
    private final AdamW.DoubleRef vAttnB;

    private final double[] dEmbWy;
    private final double[] mEmbWy;
    private final double[] vEmbWy;

    private final double[] dEmbWp;
    private final double[] mEmbWp;
    private final double[] vEmbWp;

    private final double[] dEmbB;
    private final double[] mEmbB;
    private final double[] vEmbB;

    private final StackedBiLSTM.Grad encGradAcc;

    private final LayerState[] fwdState;
    private final LayerState[] bwdState;

    private double headBiasGrad;
    private double attnBGrad;

    public RNNModelML(int inputSize, int hiddenSize) {
        this(RNNConfig.builder().inputSize(inputSize).hiddenSize(hiddenSize).build());
    }

    public RNNModelML(RNNConfig cfg) {
        this.cfg = copyConfig(cfg);
        sanitizeConfig();
        RNNConfig c = this.cfg;
        this.rng = new Random(c.seed);
        this.modelLock = new ReentrantReadWriteLock();
        this.readLock = modelLock.readLock();
        this.writeLock = modelLock.writeLock();

        this.rawPre = new RawSequencePreprocessor(c.inputSize, rng);
        this.statPre = new StatisticalSequencePreprocessor(c.inputSize);
        this.hybridPre = new HybridSequencePreprocessor(c.inputSize);
        this.activePre = pickPre(c.inputMode);

        this.encoder = new StackedBiLSTM(c.inputSize, c.hiddenSize, c.numLayers, c.bidirectional, rng);

        int outSize = encoder.outputSize();

        this.attnPooling = new AttentionPooling(outSize, rng);
        this.lastPooling = new LastHiddenPooling(outSize);
        this.meanPooling = new MeanPooling(outSize);
        this.maxPooling = new MaxPooling(outSize);
        this.activePooling = pickPool(c.poolingMode);

        this.head = new BinaryHead(outSize, rng);
        this.opt = new AdamW();

        this.step = 0L;

        this.dHeadV = new double[outSize];
        this.mHeadV = new double[outSize];
        this.vHeadV = new double[outSize];
        this.headBias = new AdamW.DoubleRef(head.bias);
        this.mHeadBias = new AdamW.DoubleRef(0.0);
        this.vHeadBias = new AdamW.DoubleRef(0.0);

        this.dAttnW = new double[outSize];
        this.mAttnW = new double[outSize];
        this.vAttnW = new double[outSize];
        this.attnB = new AdamW.DoubleRef(attnPooling.b);
        this.mAttnB = new AdamW.DoubleRef(0.0);
        this.vAttnB = new AdamW.DoubleRef(0.0);

        this.dEmbWy = new double[c.inputSize];
        this.mEmbWy = new double[c.inputSize];
        this.vEmbWy = new double[c.inputSize];

        this.dEmbWp = new double[c.inputSize];
        this.mEmbWp = new double[c.inputSize];
        this.vEmbWp = new double[c.inputSize];

        this.dEmbB = new double[c.inputSize];
        this.mEmbB = new double[c.inputSize];
        this.vEmbB = new double[c.inputSize];

        this.encGradAcc = new StackedBiLSTM.Grad(encoder);

        int L = c.numLayers;
        this.fwdState = new LayerState[L];
        this.bwdState = c.bidirectional ? new LayerState[L] : null;

        for (int l = 0; l < L; l++) {
            fwdState[l] = new LayerState(encoder.fwd[l]);
            if (c.bidirectional) bwdState[l] = new LayerState(encoder.bwd[l]);
        }
    }

    private static RNNConfig copyConfig(RNNConfig src) {
        RNNConfig in = src == null ? new RNNConfig() : src;
        RNNConfig out = new RNNConfig();
        out.inputSize = in.inputSize;
        out.hiddenSize = in.hiddenSize;
        out.numLayers = in.numLayers;
        out.bidirectional = in.bidirectional;
        out.inputMode = in.inputMode;
        out.poolingMode = in.poolingMode;
        out.learningRate = in.learningRate;
        out.dropoutRate = in.dropoutRate;
        out.recurrentDropoutRate = in.recurrentDropoutRate;
        out.weightDecay = in.weightDecay;
        out.gradientClip = in.gradientClip;
        out.labelSmoothing = in.labelSmoothing;
        out.seed = in.seed;
        return out;
    }

    private SequencePreprocessor pickPre(InputMode m) {
        if (m == InputMode.STATISTICAL_FEATURES) return statPre;
        if (m == InputMode.HYBRID) return hybridPre;
        return rawPre;
    }

    private PoolingStrategy pickPool(PoolingMode m) {
        if (m == PoolingMode.MEAN_POOLING) return meanPooling;
        if (m == PoolingMode.MAX_POOLING) return maxPooling;
        if (m == PoolingMode.ATTENTION) return attnPooling;
        return lastPooling;
    }

    public void setLearningRate(double v) {
        writeLock.lock();
        try {
            cfg.learningRate = clampFinite(v, MIN_LEARNING_RATE, MAX_LEARNING_RATE, cfg.learningRate);
        } finally {
            writeLock.unlock();
        }
    }
    public void setDropoutRate(double v) {
        writeLock.lock();
        try {
            cfg.dropoutRate = clampFinite(v, 0.0, MAX_DROPOUT, cfg.dropoutRate);
        } finally {
            writeLock.unlock();
        }
    }
    public void setRecurrentDropoutRate(double v) {
        writeLock.lock();
        try {
            cfg.recurrentDropoutRate = clampFinite(v, 0.0, MAX_DROPOUT, cfg.recurrentDropoutRate);
        } finally {
            writeLock.unlock();
        }
    }
    public void setWeightDecay(double v) {
        writeLock.lock();
        try {
            cfg.weightDecay = clampFinite(v, 0.0, MAX_WEIGHT_DECAY, cfg.weightDecay);
        } finally {
            writeLock.unlock();
        }
    }
    public void setGradientClip(double v) {
        writeLock.lock();
        try {
            cfg.gradientClip = clampFinite(v, 0.0, MAX_GRAD_CLIP, cfg.gradientClip);
        } finally {
            writeLock.unlock();
        }
    }
    public void setLabelSmoothing(double v) {
        writeLock.lock();
        try {
            cfg.labelSmoothing = clampFinite(v, 0.0, MAX_LABEL_SMOOTHING, cfg.labelSmoothing);
        } finally {
            writeLock.unlock();
        }
    }
    public void setBatchSize(int v) {
        writeLock.lock();
        try {
            this.batchSize = Math.max(1, v);
        } finally {
            writeLock.unlock();
        }
    }

    public void setInputMode(InputMode mode) {
        writeLock.lock();
        try {
            if (mode == null) mode = InputMode.HYBRID;
            cfg.inputMode = mode;
            activePre = pickPre(mode);
        } finally {
            writeLock.unlock();
        }
    }

    public void setPoolingMode(PoolingMode mode) {
        writeLock.lock();
        try {
            if (mode == null) mode = PoolingMode.ATTENTION;
            cfg.poolingMode = mode;
            activePooling = pickPool(mode);
        } finally {
            writeLock.unlock();
        }
    }

    private double[][] prepareVectors(List<ObjectML> o) {
        if (o == null || o.size() < 2) return new double[0][0];
        ObjectML yawObj = o.get(0);
        ObjectML pitchObj = o.get(1);
        if (yawObj == null || pitchObj == null) return new double[0][0];

        List<Double> yaws = yawObj.getValues();
        List<Double> pitches = pitchObj.getValues();
        if (yaws == null || pitches == null) return new double[0][0];

        int len = Math.min(yaws.size(), pitches.size());
        if (len <= 0) return new double[0][0];

        double[][] res = new double[len][2];
        for (int i = 0; i < len; i++) {
            res[i][0] = safeFinite(yaws.get(i));
            res[i][1] = safeFinite(pitches.get(i));
        }
        return res;
    }

    private static double safeFinite(Double v) {
        if (v == null) return 0.0;
        double x = v;
        return Double.isFinite(x) ? x : 0.0;
    }

    private void appendChunks(double[][] vecs, double label, List<Sample> out) {
        for (int i = 0; i < vecs.length; i += DEFAULT_CHUNK_SIZE) {
            int end = Math.min(vecs.length, i + DEFAULT_CHUNK_SIZE);
            if (end - i < 2) continue;
            double[][] chunk = new double[end - i][2];
            System.arraycopy(vecs, i, chunk, 0, end - i);
            out.add(new Sample(chunk, label));
        }
    }

    @Override
    public ResultML checkData(List<ObjectML> o) {
        readLock.lock();
        try {
            ResultML r = new ResultML();
            double prob = 0.5;
            if (o == null || o.isEmpty()) {
                Logger.warn("checkData received empty input; returning neutral probability 0.5");
            } else {
                double[][] vecs = prepareVectors(o);
                if (vecs.length >= 2) {
                    prob = forwardProbabilityOrNaN(vecs);
                    if (!Double.isFinite(prob)) {
                        Logger.warn("checkData produced non-finite probability; returning neutral probability 0.5");
                        prob = 0.5;
                    }
                } else {
                    Logger.warn("checkData received sequence with less than 2 valid points; returning neutral probability 0.5");
                }
            }
            if (!Double.isFinite(prob)) {
                Logger.warn("checkData returned non-finite probability after sanitization; forcing 0.5");
                prob = 0.5;
            }
            prob = Math.max(0.0, Math.min(1.0, prob));

            r.statisticsResult.UNUSUAL = prob;
            r.statisticsResult.STRANGE = 0;
            r.statisticsResult.SUSPECTED = 0;
            r.statisticsResult.SUSPICIOUSLY = 0;

            return r;
        } finally {
            readLock.unlock();
        }
    }

    @Override
    public void learnByData(List<ObjectML> o, boolean isMustBeBlocked) {
        writeLock.lock();
        try {
            if (o == null || o.isEmpty()) return;

            double y = isMustBeBlocked ? 1.0 : 0.0;
            if (cfg.labelSmoothing > 0.0) y = y * (1.0 - cfg.labelSmoothing) + 0.5 * cfg.labelSmoothing;

            double[][] vecs = prepareVectors(o);
            if (vecs.length < 2) return;

            List<Sample> samples = new ArrayList<>();
            appendChunks(vecs, y, samples);
            if (samples.isEmpty()) return;

            int bs = Math.max(1, batchSize);
            int skippedInvalid = 0;
            int skippedNonFiniteProb = 0;
            List<Sample> currentBatch = new ArrayList<>(bs);
            for (Sample s : samples) {
                currentBatch.add(s);
                if (currentBatch.size() >= bs) {
                    TrainBatchResult res = trainBatch(currentBatch);
                    skippedInvalid += res.skippedInvalidSequence;
                    skippedNonFiniteProb += res.skippedNonFiniteProb;
                    currentBatch.clear();
                }
            }
            if (!currentBatch.isEmpty()) {
                TrainBatchResult res = trainBatch(currentBatch);
                skippedInvalid += res.skippedInvalidSequence;
                skippedNonFiniteProb += res.skippedNonFiniteProb;
            }
            if (skippedInvalid > 0 || skippedNonFiniteProb > 0) {
                Logger.warn(String.format(
                                "learnByData skipped samples -> invalid sequence: %d, non-finite probability: %d",
                                skippedInvalid, skippedNonFiniteProb));
            }
        } finally {
            writeLock.unlock();
        }
    }

    private TrainBatchResult trainBatch(List<Sample> batch) {
        TrainBatchResult res = new TrainBatchResult();
        zeroBatchGrads();

        int used = 0;
        int embeddingUsed = 0;
        double batchLoss = 0.0;
        double correct = 0.0;
        boolean attentionActive = activePooling == attnPooling;

        for (Sample s : batch) {
            ForwardCache fc = forwardCache(s.vecs);
            if (fc == null) {
                res.skippedInvalidSequence++;
                continue;
            }

            double p = fc.prob;
            if (!Double.isFinite(p)) {
                res.skippedNonFiniteProb++;
                continue;
            }

            boolean predictedCheat = p >= 0.5;
            boolean actualCheat = s.label >= 0.5;
            if (predictedCheat == actualCheat) {
                correct += 1.0;
            }

            p = Math.max(1e-15, Math.min(1.0 - 1e-15, p));
            double loss = - (s.label * Math.log(p) + (1.0 - s.label) * Math.log(1.0 - p));
            batchLoss += loss;

            double dLogit = (p - s.label);

            BinaryHead.Grad hg = new BinaryHead.Grad(head.in);
            double[] dPooled = head.backward(fc.headCache, dLogit, hg);
            add(dHeadV, hg.dV);
            headBiasGrad += hg.dBias;

            PoolingGrad pg = attentionActive ? new PoolingGrad() : null;
            if (attentionActive) {
                pg.dAttentionW = dAttnW;
                pg.dAttentionB = 0.0;
            }
            double[][] dH = activePooling.backward(fc.hTime, fc.seq.mask, dPooled, fc.poolCache, pg);
            if (attentionActive) {
                attnBGrad += pg.dAttentionB;
            }

            double[][] dX = encoder.backward(fc.encCache, dH, encGradAcc);

            if (fc.usesRawEmbedding) {
                accumulateEmbeddingGrad(s.vecs, dX);
                embeddingUsed++;
            }

            used++;
        }

        if (used > 0) {
            opt.incrementStep();
            applyUpdate(used, embeddingUsed, attentionActive);
            step++;
            res.loss = batchLoss;
            res.correct = correct;
            res.used = used;
            return res;
        }

        return res;
    }

    private ValidationResult validateBatch(List<Sample> batch) {
        ValidationResult res = new ValidationResult();
        for (Sample s : batch) {
            double p = forwardProbabilityOrNaN(s.vecs);
            if (!Double.isFinite(p)) {
                res.skippedInvalidOrNonFinite++;
                continue;
            }
            p = Math.max(1e-15, Math.min(1.0 - 1e-15, p));
            double loss = - (s.label * Math.log(p) + (1.0 - s.label) * Math.log(1.0 - p));

            res.loss += loss;
            res.pairs.add(new PredictionPair(p, s.label));
            res.used++;
        }
        return res;
    }

    private ForwardCache forwardCache(double[][] raw) {
        SequenceData seq = activePre.prepare(raw);
        if (!hasEnoughValidSteps(seq, 2)) return null;

        StackedBiLSTM.Cache encCache = new StackedBiLSTM.Cache();
        double[][] hTime = encoder.forward(seq.x, true, cfg.dropoutRate, cfg.recurrentDropoutRate, rng, encCache);

        PoolingCache pc = new PoolingCache();
        double[] pooled = activePooling.forward(hTime, seq.mask, pc);

        BinaryHead.Cache hc = new BinaryHead.Cache();
        double prob = head.forward(pooled, hc);

        ForwardCache fc = new ForwardCache();
        fc.seq = seq;
        fc.hTime = hTime;
        fc.encCache = encCache;
        fc.poolCache = pc;
        fc.headCache = hc;
        fc.prob = prob;
        fc.usesRawEmbedding = usesRawEmbedding();
        return fc;
    }

    private double forwardProbability(double[][] raw) {
        return forwardProbability(raw, 0.5);
    }

    private double forwardProbabilityOrNaN(double[][] raw) {
        return forwardProbability(raw, Double.NaN);
    }

    private double forwardProbability(double[][] raw, double fallback) {
        if (raw == null || raw.length < 2) return fallback;
        double sum = 0.0;
        int used = 0;

        for (int i = 0; i < raw.length; i += DEFAULT_CHUNK_SIZE) {
            int end = Math.min(raw.length, i + DEFAULT_CHUNK_SIZE);
            if (end - i < 2) continue;
            double[][] chunk = new double[end - i][2];
            System.arraycopy(raw, i, chunk, 0, end - i);
            double p = forwardProbabilitySingleChunk(chunk, Double.NaN);
            if (!Double.isFinite(p)) continue;
            sum += p;
            used++;
        }

        if (used <= 0) return fallback;
        double p = sum / used;
        if (!Double.isFinite(p)) return fallback;
        return Math.max(0.0, Math.min(1.0, p));
    }

    private double forwardProbabilitySingleChunk(double[][] raw, double fallback) {
        SequenceData seq = activePre.prepare(raw);
        if (!hasEnoughValidSteps(seq, 2)) return fallback;

        double[][] hTime = encoder.forward(seq.x, false, 0.0, 0.0, rng, null);
        double[] pooled = activePooling.forward(hTime, seq.mask, null);
        double p = head.forward(pooled, null);
        if (!Double.isFinite(p)) return fallback;
        return Math.max(0.0, Math.min(1.0, p));
    }

    private void accumulateEmbeddingGrad(double[][] raw, double[][] dX) {
        int T = Math.min(raw.length, dX.length);
        for (int t = 0; t < T; t++) {
            double vY = Math.tanh(raw[t][0] / 100.0);
            double vP = Math.tanh(raw[t][1] / 100.0);
            for (int i = 0; i < cfg.inputSize; i++) {
                double g = dX[t][i];
                dEmbWy[i] += g * vY;
                dEmbWp[i] += g * vP;
                dEmbB[i] += g;
            }
        }
    }

    private void applyUpdate(int used, int embeddingUsed, boolean updateAttention) {
        double inv = 1.0 / used;
        boolean updateEmbedding = embeddingUsed > 0;
        double embInv = updateEmbedding ? (1.0 / embeddingUsed) : 0.0;

        scaleInPlace(dHeadV, inv);
        headBiasGrad *= inv;

        if (updateAttention) {
            scaleInPlace(dAttnW, inv);
            attnBGrad *= inv;
        }

        if (updateEmbedding) {
            scaleInPlace(dEmbWy, embInv);
            scaleInPlace(dEmbWp, embInv);
            scaleInPlace(dEmbB, embInv);
        }

        for (int l = 0; l < cfg.numLayers; l++) {
            scaleLayerGrad(encGradAcc.fwd[l], inv);
            if (cfg.bidirectional) scaleLayerGrad(encGradAcc.bwd[l], inv);
        }

        if (cfg.gradientClip > 0.0 && Double.isFinite(cfg.gradientClip)) {
            double sq = 0.0;
            sq += squareNorm(dHeadV) + headBiasGrad * headBiasGrad;
            if (updateAttention) {
                sq += squareNorm(dAttnW) + attnBGrad * attnBGrad;
            }
            if (updateEmbedding) {
                sq += squareNorm(dEmbWy);
                sq += squareNorm(dEmbWp);
                sq += squareNorm(dEmbB);
            }
            for (int l = 0; l < cfg.numLayers; l++) {
                sq += squareNorm(encGradAcc.fwd[l]);
                if (cfg.bidirectional) sq += squareNorm(encGradAcc.bwd[l]);
            }

            double norm = Math.sqrt(sq);
            if (norm > cfg.gradientClip) {
                double s = cfg.gradientClip / (norm + 1e-12);
                scaleInPlace(dHeadV, s);
                headBiasGrad *= s;
                if (updateAttention) {
                    scaleInPlace(dAttnW, s);
                    attnBGrad *= s;
                }
                if (updateEmbedding) {
                    scaleInPlace(dEmbWy, s);
                    scaleInPlace(dEmbWp, s);
                    scaleInPlace(dEmbB, s);
                }
                for (int l = 0; l < cfg.numLayers; l++) {
                    scaleLayerGrad(encGradAcc.fwd[l], s);
                    if (cfg.bidirectional) scaleLayerGrad(encGradAcc.bwd[l], s);
                }
            }
        }

        headBias.value = head.bias;
        opt.step(head.V, dHeadV, mHeadV, vHeadV, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.stepScalar(headBias, headBiasGrad, mHeadBias, vHeadBias, cfg.learningRate, 0.0, 0.0);
        head.bias = headBias.value;

        if (updateAttention) {
            attnB.value = attnPooling.b;
            opt.step(attnPooling.W, dAttnW, mAttnW, vAttnW, cfg.learningRate, cfg.weightDecay, 0.0);
            opt.stepScalar(attnB, attnBGrad, mAttnB, vAttnB, cfg.learningRate, 0.0, 0.0);
            attnPooling.b = attnB.value;
        }

        if (updateEmbedding) {
            opt.step(rawPre.getEmbeddingWy(), dEmbWy, mEmbWy, vEmbWy, cfg.learningRate, cfg.weightDecay, 0.0);
            opt.step(rawPre.getEmbeddingWp(), dEmbWp, mEmbWp, vEmbWp, cfg.learningRate, cfg.weightDecay, 0.0);
            opt.step(rawPre.getEmbeddingB(), dEmbB, mEmbB, vEmbB, cfg.learningRate, cfg.weightDecay, 0.0);
        }

        for (int l = 0; l < cfg.numLayers; l++) {
            updateLayer(encoder.fwd[l], encGradAcc.fwd[l], fwdState[l]);
            if (cfg.bidirectional) updateLayer(encoder.bwd[l], encGradAcc.bwd[l], bwdState[l]);
        }
    }

    private boolean usesRawEmbedding() {
        return activePre == rawPre;
    }

    private void updateLayer(LSTMLayer layer, LSTMLayer.Grad g, LayerState s) {
        opt.step(layer.Wf, g.dWf, s.mWf, s.vWf, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.step(layer.Wi, g.dWi, s.mWi, s.vWi, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.step(layer.Wc, g.dWc, s.mWc, s.vWc, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.step(layer.Wo, g.dWo, s.mWo, s.vWo, cfg.learningRate, cfg.weightDecay, 0.0);

        opt.step(layer.Uf, g.dUf, s.mUf, s.vUf, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.step(layer.Ui, g.dUi, s.mUi, s.vUi, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.step(layer.Uc, g.dUc, s.mUc, s.vUc, cfg.learningRate, cfg.weightDecay, 0.0);
        opt.step(layer.Uo, g.dUo, s.mUo, s.vUo, cfg.learningRate, cfg.weightDecay, 0.0);

        opt.step(layer.bf, g.dbf, s.mbf, s.vbf, cfg.learningRate, 0.0, 0.0);
        opt.step(layer.bi, g.dbi, s.mbi, s.vbi, cfg.learningRate, 0.0, 0.0);
        opt.step(layer.bc, g.dbc, s.mbc, s.vbc, cfg.learningRate, 0.0, 0.0);
        opt.step(layer.bo, g.dbo, s.mbo, s.vbo, cfg.learningRate, 0.0, 0.0);

        opt.step(layer.lnGamma, g.dLnGamma, s.mLnG, s.vLnG, cfg.learningRate, 0.0, 0.0);
        opt.step(layer.lnBeta, g.dLnBeta, s.mLnB, s.vLnB, cfg.learningRate, 0.0, 0.0);
    }

    private void zeroBatchGrads() {
        zero(dHeadV);
        headBiasGrad = 0.0;

        zero(dAttnW);
        attnBGrad = 0.0;

        zero(dEmbWy);
        zero(dEmbWp);
        zero(dEmbB);

        for (int l = 0; l < encGradAcc.fwd.length; l++) {
            encGradAcc.fwd[l].zero();
            if (cfg.bidirectional) encGradAcc.bwd[l].zero();
        }
    }

    private static void zero(double[] a) { for (int i = 0; i < a.length; i++) a[i] = 0.0; }

    private static void add(double[] dst, double[] src) {
        for (int i = 0; i < dst.length; i++) dst[i] += src[i];
    }

    private static void scaleInPlace(double[] a, double s) {
        for (int i = 0; i < a.length; i++) a[i] *= s;
    }

    private static void scaleLayerGrad(LSTMLayer.Grad g, double s) {
        scaleInPlace(g.dWf, s); scaleInPlace(g.dWi, s); scaleInPlace(g.dWc, s); scaleInPlace(g.dWo, s);
        scaleInPlace(g.dUf, s); scaleInPlace(g.dUi, s); scaleInPlace(g.dUc, s); scaleInPlace(g.dUo, s);
        scaleInPlace(g.dbf, s); scaleInPlace(g.dbi, s); scaleInPlace(g.dbc, s); scaleInPlace(g.dbo, s);
        scaleInPlace(g.dLnGamma, s); scaleInPlace(g.dLnBeta, s);
    }

    private static double squareNorm(double[] a) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) s += a[i] * a[i];
        return s;
    }

    private static double squareNorm(LSTMLayer.Grad g) {
        double s = 0.0;
        s += squareNorm(g.dWf); s += squareNorm(g.dWi); s += squareNorm(g.dWc); s += squareNorm(g.dWo);
        s += squareNorm(g.dUf); s += squareNorm(g.dUi); s += squareNorm(g.dUc); s += squareNorm(g.dUo);
        s += squareNorm(g.dbf); s += squareNorm(g.dbi); s += squareNorm(g.dbc); s += squareNorm(g.dbo);
        s += squareNorm(g.dLnGamma); s += squareNorm(g.dLnBeta);
        return s;
    }

    private static boolean hasEnoughValidSteps(SequenceData seq, int minSteps) {
        if (seq == null || seq.x == null || seq.mask == null || minSteps <= 0) return false;
        int len = Math.min(seq.x.length, seq.mask.length);
        if (len < minSteps) return false;
        int valid = 0;
        for (int i = 0; i < len; i++) {
            if (seq.mask[i] > 0.5 && seq.x[i] != null) {
                valid++;
                if (valid >= minSteps) return true;
            }
        }
        return false;
    }

    private static double clampFinite(double v, double min, double max, double fallback) {
        if (!Double.isFinite(v)) return fallback;
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    private static int clampInt(int v, int min, int max) {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    private void sanitizeConfig() {
        cfg.inputSize = clampInt(cfg.inputSize, MIN_INPUT_SIZE, MAX_INPUT_SIZE);
        cfg.hiddenSize = clampInt(cfg.hiddenSize, MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE);
        cfg.numLayers = clampInt(cfg.numLayers, MIN_NUM_LAYERS, MAX_NUM_LAYERS);

        if (cfg.inputMode == null) cfg.inputMode = InputMode.HYBRID;
        if (cfg.poolingMode == null) cfg.poolingMode = PoolingMode.ATTENTION;

        cfg.learningRate = clampFinite(cfg.learningRate, MIN_LEARNING_RATE, MAX_LEARNING_RATE, DEFAULT_LEARNING_RATE);
        cfg.dropoutRate = clampFinite(cfg.dropoutRate, 0.0, MAX_DROPOUT, DEFAULT_DROPOUT);
        cfg.recurrentDropoutRate = clampFinite(cfg.recurrentDropoutRate, 0.0, MAX_DROPOUT, DEFAULT_RECURRENT_DROPOUT);
        cfg.weightDecay = clampFinite(cfg.weightDecay, 0.0, MAX_WEIGHT_DECAY, DEFAULT_WEIGHT_DECAY);
        cfg.gradientClip = clampFinite(cfg.gradientClip, 0.0, MAX_GRAD_CLIP, DEFAULT_GRAD_CLIP);
        cfg.labelSmoothing = clampFinite(cfg.labelSmoothing, 0.0, MAX_LABEL_SMOOTHING, DEFAULT_LABEL_SMOOTHING);
    }

    @Override
    public void saveToFile(String fileName) {
        readLock.lock();
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(Paths.get(fileName))))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);

            out.writeInt(cfg.inputSize);
            out.writeInt(cfg.hiddenSize);
            out.writeInt(cfg.numLayers);
            out.writeBoolean(cfg.bidirectional);

            out.writeInt(cfg.inputMode.ordinal());
            out.writeInt(cfg.poolingMode.ordinal());

            out.writeDouble(cfg.learningRate);
            out.writeDouble(cfg.dropoutRate);
            out.writeDouble(cfg.recurrentDropoutRate);
            out.writeDouble(cfg.weightDecay);
            out.writeDouble(cfg.gradientClip);
            out.writeDouble(cfg.labelSmoothing);

            out.writeInt(batchSize);
            out.writeLong(step);
            out.writeLong(opt.t);

            ModelIO.writeArr(out, rawPre.getEmbeddingWy());
            ModelIO.writeArr(out, rawPre.getEmbeddingWp());
            ModelIO.writeArr(out, rawPre.getEmbeddingB());

            writeEncoder(out);
            ModelIO.writeArr(out, attnPooling.W);
            out.writeDouble(attnPooling.b);

            ModelIO.writeArr(out, head.V);
            out.writeDouble(head.bias);

            writeOptimizerState(out);
        } catch (Exception e) {
            Logger.error("Failed to save RNN model to " + fileName + ": " + e.getMessage());
            throw new IllegalStateException("Failed to save RNN model to " + fileName, e);
        } finally {
            readLock.unlock();
        }
    }

    public void load(InputStream in) {
        writeLock.lock();
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(in))) {
            int magic = dis.readInt();
            int ver = dis.readInt();
            if (magic != MAGIC) throw new IllegalStateException("bad model magic");
            if (ver != VERSION && ver != LEGACY_VERSION) throw new IllegalStateException("bad model version: " + ver);

            int inSize = dis.readInt();
            int hid = dis.readInt();
            int layers = dis.readInt();
            boolean bi = dis.readBoolean();

            if (inSize != cfg.inputSize || hid != cfg.hiddenSize || layers != cfg.numLayers || bi != cfg.bidirectional) {
                throw new IllegalStateException("arch mismatch");
            }

            int im = dis.readInt();
            int pm = dis.readInt();
            cfg.inputMode = InputMode.values()[clamp(im, InputMode.values().length)];
            cfg.poolingMode = PoolingMode.values()[clamp(pm, PoolingMode.values().length)];
            activePre = pickPre(cfg.inputMode);
            activePooling = pickPool(cfg.poolingMode);

            cfg.learningRate = dis.readDouble();
            cfg.dropoutRate = dis.readDouble();
            cfg.recurrentDropoutRate = dis.readDouble();
            cfg.weightDecay = dis.readDouble();
            cfg.gradientClip = dis.readDouble();
            cfg.labelSmoothing = dis.readDouble();
            sanitizeConfig();

            batchSize = Math.max(1, dis.readInt());
            step = dis.readLong();
            opt.t = dis.readLong();

            readInto(dis, rawPre.getEmbeddingWy());
            readInto(dis, rawPre.getEmbeddingWp());
            readInto(dis, rawPre.getEmbeddingB());

            readEncoder(dis);

            readInto(dis, attnPooling.W);
            attnPooling.b = dis.readDouble();

            readInto(dis, head.V);
            head.bias = dis.readDouble();

            if (ver >= VERSION) {
                readOptimizerState(dis);
            } else {
                resetOptimizerState();
            }
        } catch (Exception e) {
            Logger.error("Failed to load RNN model: " + e.getMessage());
            throw new IllegalStateException("Failed to load RNN model", e);
        } finally {
            writeLock.unlock();
        }
    }

    private static int clamp(int v, int n) {
        if (n <= 0) return 0;
        if (v < 0) return 0;
        if (v >= n) return n - 1;
        return v;
    }

    private void writeEncoder(DataOutputStream out) throws Exception {
        out.writeInt(cfg.numLayers);
        out.writeBoolean(cfg.bidirectional);

        for (int l = 0; l < cfg.numLayers; l++) {
            writeLayer(out, encoder.fwd[l]);
            if (cfg.bidirectional) writeLayer(out, encoder.bwd[l]);
        }
    }

    private void readEncoder(DataInputStream in) throws Exception {
        int layers = in.readInt();
        boolean bi = in.readBoolean();
        if (layers != cfg.numLayers || bi != cfg.bidirectional) throw new IllegalStateException("encoder mismatch");

        for (int l = 0; l < cfg.numLayers; l++) {
            readLayer(in, encoder.fwd[l]);
            if (cfg.bidirectional) readLayer(in, encoder.bwd[l]);
        }
    }

    private void writeLayer(DataOutputStream out, LSTMLayer l) throws Exception {
        ModelIO.writeArr(out, l.Wf); ModelIO.writeArr(out, l.Wi); ModelIO.writeArr(out, l.Wc); ModelIO.writeArr(out, l.Wo);
        ModelIO.writeArr(out, l.Uf); ModelIO.writeArr(out, l.Ui); ModelIO.writeArr(out, l.Uc); ModelIO.writeArr(out, l.Uo);
        ModelIO.writeArr(out, l.bf); ModelIO.writeArr(out, l.bi); ModelIO.writeArr(out, l.bc); ModelIO.writeArr(out, l.bo);
        ModelIO.writeArr(out, l.lnGamma); ModelIO.writeArr(out, l.lnBeta);
    }

    private void readLayer(DataInputStream in, LSTMLayer l) throws Exception {
        readInto(in, l.Wf); readInto(in, l.Wi); readInto(in, l.Wc); readInto(in, l.Wo);
        readInto(in, l.Uf); readInto(in, l.Ui); readInto(in, l.Uc); readInto(in, l.Uo);
        readInto(in, l.bf); readInto(in, l.bi); readInto(in, l.bc); readInto(in, l.bo);
        readInto(in, l.lnGamma); readInto(in, l.lnBeta);
    }

    private static void readInto(DataInputStream in, double[] dst) throws Exception {
        double[] a = ModelIO.readArr(in);
        if (a == null) {
            throw new IllegalStateException("missing array in model file");
        }
        if (a.length != dst.length) {
            throw new IllegalStateException("array length mismatch: expected " + dst.length + ", got " + a.length);
        }
        for (double v : a) {
            if (!Double.isFinite(v)) {
                throw new IllegalStateException("model file contains non-finite weight");
            }
        }
        System.arraycopy(a, 0, dst, 0, dst.length);
    }

    private void writeOptimizerState(DataOutputStream out) throws Exception {
        ModelIO.writeArr(out, mHeadV);
        ModelIO.writeArr(out, vHeadV);
        out.writeDouble(mHeadBias.value);
        out.writeDouble(vHeadBias.value);

        ModelIO.writeArr(out, mAttnW);
        ModelIO.writeArr(out, vAttnW);
        out.writeDouble(mAttnB.value);
        out.writeDouble(vAttnB.value);

        ModelIO.writeArr(out, mEmbWy);
        ModelIO.writeArr(out, vEmbWy);
        ModelIO.writeArr(out, mEmbWp);
        ModelIO.writeArr(out, vEmbWp);
        ModelIO.writeArr(out, mEmbB);
        ModelIO.writeArr(out, vEmbB);

        for (int l = 0; l < cfg.numLayers; l++) {
            writeLayerState(out, fwdState[l]);
            if (cfg.bidirectional) writeLayerState(out, bwdState[l]);
        }
    }

    private void readOptimizerState(DataInputStream in) throws Exception {
        readInto(in, mHeadV);
        readInto(in, vHeadV);
        mHeadBias.value = in.readDouble();
        vHeadBias.value = in.readDouble();

        readInto(in, mAttnW);
        readInto(in, vAttnW);
        mAttnB.value = in.readDouble();
        vAttnB.value = in.readDouble();

        readInto(in, mEmbWy);
        readInto(in, vEmbWy);
        readInto(in, mEmbWp);
        readInto(in, vEmbWp);
        readInto(in, mEmbB);
        readInto(in, vEmbB);

        for (int l = 0; l < cfg.numLayers; l++) {
            readLayerState(in, fwdState[l]);
            if (cfg.bidirectional) readLayerState(in, bwdState[l]);
        }
    }

    private void writeLayerState(DataOutputStream out, LayerState s) throws Exception {
        ModelIO.writeArr(out, s.mWf); ModelIO.writeArr(out, s.vWf);
        ModelIO.writeArr(out, s.mWi); ModelIO.writeArr(out, s.vWi);
        ModelIO.writeArr(out, s.mWc); ModelIO.writeArr(out, s.vWc);
        ModelIO.writeArr(out, s.mWo); ModelIO.writeArr(out, s.vWo);

        ModelIO.writeArr(out, s.mUf); ModelIO.writeArr(out, s.vUf);
        ModelIO.writeArr(out, s.mUi); ModelIO.writeArr(out, s.vUi);
        ModelIO.writeArr(out, s.mUc); ModelIO.writeArr(out, s.vUc);
        ModelIO.writeArr(out, s.mUo); ModelIO.writeArr(out, s.vUo);

        ModelIO.writeArr(out, s.mbf); ModelIO.writeArr(out, s.vbf);
        ModelIO.writeArr(out, s.mbi); ModelIO.writeArr(out, s.vbi);
        ModelIO.writeArr(out, s.mbc); ModelIO.writeArr(out, s.vbc);
        ModelIO.writeArr(out, s.mbo); ModelIO.writeArr(out, s.vbo);

        ModelIO.writeArr(out, s.mLnG); ModelIO.writeArr(out, s.vLnG);
        ModelIO.writeArr(out, s.mLnB); ModelIO.writeArr(out, s.vLnB);
    }

    private void readLayerState(DataInputStream in, LayerState s) throws Exception {
        readInto(in, s.mWf); readInto(in, s.vWf);
        readInto(in, s.mWi); readInto(in, s.vWi);
        readInto(in, s.mWc); readInto(in, s.vWc);
        readInto(in, s.mWo); readInto(in, s.vWo);

        readInto(in, s.mUf); readInto(in, s.vUf);
        readInto(in, s.mUi); readInto(in, s.vUi);
        readInto(in, s.mUc); readInto(in, s.vUc);
        readInto(in, s.mUo); readInto(in, s.vUo);

        readInto(in, s.mbf); readInto(in, s.vbf);
        readInto(in, s.mbi); readInto(in, s.vbi);
        readInto(in, s.mbc); readInto(in, s.vbc);
        readInto(in, s.mbo); readInto(in, s.vbo);

        readInto(in, s.mLnG); readInto(in, s.vLnG);
        readInto(in, s.mLnB); readInto(in, s.vLnB);
    }

    private void resetOptimizerState() {
        zero(mHeadV); zero(vHeadV);
        mHeadBias.value = 0.0; vHeadBias.value = 0.0;

        zero(mAttnW); zero(vAttnW);
        mAttnB.value = 0.0; vAttnB.value = 0.0;

        zero(mEmbWy); zero(vEmbWy);
        zero(mEmbWp); zero(vEmbWp);
        zero(mEmbB); zero(vEmbB);

        for (int l = 0; l < cfg.numLayers; l++) {
            zeroLayerState(fwdState[l]);
            if (cfg.bidirectional) zeroLayerState(bwdState[l]);
        }
        opt.t = 0L;
    }

    private static void zeroLayerState(LayerState s) {
        zero(s.mWf); zero(s.vWf); zero(s.mWi); zero(s.vWi); zero(s.mWc); zero(s.vWc); zero(s.mWo); zero(s.vWo);
        zero(s.mUf); zero(s.vUf); zero(s.mUi); zero(s.vUi); zero(s.mUc); zero(s.vUc); zero(s.mUo); zero(s.vUo);
        zero(s.mbf); zero(s.vbf); zero(s.mbi); zero(s.vbi); zero(s.mbc); zero(s.vbc); zero(s.mbo); zero(s.vbo);
        zero(s.mLnG); zero(s.vLnG); zero(s.mLnB); zero(s.vLnB);
    }

    @Override
    public int parameters() {
        readLock.lock();
        try {
            long p = 0;

            p += rawPre.getEmbeddingWy().length;
            p += rawPre.getEmbeddingWp().length;
            p += rawPre.getEmbeddingB().length;

            for (int l = 0; l < cfg.numLayers; l++) {
                p += countLayer(encoder.fwd[l]);
                if (cfg.bidirectional) p += countLayer(encoder.bwd[l]);
            }

            p += attnPooling.W.length + 1;
            p += head.V.length + 1;

            if (p > Integer.MAX_VALUE) return Integer.MAX_VALUE;
            return (int) p;
        } finally {
            readLock.unlock();
        }
    }

    private static long countLayer(LSTMLayer l) {
        long p = 0;
        p += l.Wf.length + l.Wi.length + l.Wc.length + l.Wo.length;
        p += l.Uf.length + l.Ui.length + l.Uc.length + l.Uo.length;
        p += l.bf.length + l.bi.length + l.bc.length + l.bo.length;
        p += l.lnGamma.length + l.lnBeta.length;
        return p;
    }

    private static final class Sample {
        final double[][] vecs;
        final double label;
        Sample(double[][] vecs, double label) {
            this.vecs = vecs;
            this.label = label;
        }
    }

    private static final class PredictionPair implements Comparable<PredictionPair> {
        final double prob;
        final double label;
        PredictionPair(double prob, double label) {
            this.prob = prob;
            this.label = label;
        }
        @Override
        public int compareTo(PredictionPair o) {
            return Double.compare(o.prob, this.prob);
        }
    }

    private static final class ValidationResult {
        double loss;
        int used;
        int skippedInvalidOrNonFinite;
        final List<PredictionPair> pairs = new ArrayList<>();
    }

    private static final class DatasetMetrics {
        double lossSum;
        double avgLoss;
        double acc;
        int used;
        int skippedInvalidOrNonFinite;
        int tp;
        int tn;
        int fp;
        int fn;
        final List<PredictionPair> pairs = new ArrayList<>();
    }

    private static final class TrainBatchResult {
        double loss;
        double correct;
        int used;
        int skippedInvalidSequence;
        int skippedNonFiniteProb;
    }

    private static final class ForwardCache {
        SequenceData seq;
        double[][] hTime;
        StackedBiLSTM.Cache encCache;
        PoolingCache poolCache;
        BinaryHead.Cache headCache;
        double prob;
        boolean usesRawEmbedding;
    }

    private static final class LayerState {
        final double[] mWf, vWf, mWi, vWi, mWc, vWc, mWo, vWo;
        final double[] mUf, vUf, mUi, vUi, mUc, vUc, mUo, vUo;
        final double[] mbf, vbf, mbi, vbi, mbc, vbc, mbo, vbo;
        final double[] mLnG, vLnG, mLnB, vLnB;

        LayerState(LSTMLayer l) {
            mWf = new double[l.Wf.length]; vWf = new double[l.Wf.length];
            mWi = new double[l.Wi.length]; vWi = new double[l.Wi.length];
            mWc = new double[l.Wc.length]; vWc = new double[l.Wc.length];
            mWo = new double[l.Wo.length]; vWo = new double[l.Wo.length];

            mUf = new double[l.Uf.length]; vUf = new double[l.Uf.length];
            mUi = new double[l.Ui.length]; vUi = new double[l.Ui.length];
            mUc = new double[l.Uc.length]; vUc = new double[l.Uc.length];
            mUo = new double[l.Uo.length]; vUo = new double[l.Uo.length];

            mbf = new double[l.bf.length]; vbf = new double[l.bf.length];
            mbi = new double[l.bi.length]; vbi = new double[l.bi.length];
            mbc = new double[l.bc.length]; vbc = new double[l.bc.length];
            mbo = new double[l.bo.length]; vbo = new double[l.bo.length];

            mLnG = new double[l.lnGamma.length]; vLnG = new double[l.lnGamma.length];
            mLnB = new double[l.lnBeta.length]; vLnB = new double[l.lnBeta.length];
        }
    }

    private double rocAuc(List<PredictionPair> pairs) {
        if (pairs == null || pairs.isEmpty()) return 0.0;
        List<PredictionPair> sorted = new ArrayList<>(pairs);
        sorted.sort((a, b) -> Double.compare(a.prob, b.prob));

        long pos = 0;
        long neg = 0;
        double sumPosRanks = 0.0;
        int i = 0;
        while (i < sorted.size()) {
            int j = i;
            double score = sorted.get(i).prob;
            while (j + 1 < sorted.size() && Double.compare(sorted.get(j + 1).prob, score) == 0) j++;

            double avgRank = ((double) (i + 1) + (double) (j + 1)) * 0.5;
            for (int k = i; k <= j; k++) {
                PredictionPair p = sorted.get(k);
                if (p.label >= 0.5) {
                    pos++;
                    sumPosRanks += avgRank;
                } else {
                    neg++;
                }
            }
            i = j + 1;
        }

        if (pos == 0 || neg == 0) return 0.0;
        double auc = (sumPosRanks - ((double) pos * (pos + 1) * 0.5)) / ((double) pos * neg);
        if (!Double.isFinite(auc)) return 0.0;
        if (auc < 0.0) return 0.0;
        if (auc > 1.0) return 1.0;
        return auc;
    }

    private double prAuc(List<PredictionPair> pairs) {
        if (pairs == null || pairs.isEmpty()) return 0.0;
        List<PredictionPair> sorted = new ArrayList<>(pairs);
        sorted.sort((a, b) -> Double.compare(b.prob, a.prob));

        long pos = 0;
        for (PredictionPair p : sorted) {
            if (p.label >= 0.5) pos++;
        }
        if (pos == 0) return 0.0;

        long tp = 0;
        long fp = 0;
        double auc = 0.0;
        double prevRecall = 0.0;
        double prevPrecision = 1.0;

        int i = 0;
        while (i < sorted.size()) {
            double score = sorted.get(i).prob;
            long grpPos = 0;
            long grpNeg = 0;
            while (i < sorted.size() && Double.compare(sorted.get(i).prob, score) == 0) {
                if (sorted.get(i).label >= 0.5) grpPos++;
                else grpNeg++;
                i++;
            }

            tp += grpPos;
            fp += grpNeg;
            double recall = (double) tp / pos;
            double precision = (tp + fp) == 0 ? 1.0 : (double) tp / (tp + fp);
            auc += (recall - prevRecall) * (precision + prevPrecision) * 0.5;
            prevRecall = recall;
            prevPrecision = precision;
        }

        if (!Double.isFinite(auc)) return 0.0;
        if (auc < 0.0) return 0.0;
        if (auc > 1.0) return 1.0;
        return auc;
    }

    private DatasetMetrics evaluateDataset(List<Pair<List<ObjectML>, Boolean>> dataset) {
        DatasetMetrics metrics = new DatasetMetrics();
        if (dataset == null || dataset.isEmpty()) return metrics;

        for (Pair<List<ObjectML>, Boolean> dataPair : dataset) {
            double[][] vecs = prepareVectors(dataPair.getX());
            if (vecs.length < 2) {
                metrics.skippedInvalidOrNonFinite++;
                continue;
            }

            double y = dataPair.getY() ? 1.0 : 0.0;
            double p = forwardProbabilityOrNaN(vecs);
            if (!Double.isFinite(p)) {
                metrics.skippedInvalidOrNonFinite++;
                continue;
            }

            p = Math.max(1e-15, Math.min(1.0 - 1e-15, p));
            metrics.lossSum += - (y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p));
            metrics.pairs.add(new PredictionPair(p, y));

            boolean actual = y >= 0.5;
            boolean pred = p >= 0.5;
            if (actual && pred) metrics.tp++;
            else if (!actual && !pred) metrics.tn++;
            else if (!actual) metrics.fp++;
            else metrics.fn++;

            metrics.used++;
        }

        if (metrics.used > 0) {
            metrics.avgLoss = metrics.lossSum / metrics.used;
            metrics.acc = ((double) (metrics.tp + metrics.tn) / metrics.used) * 100.0;
        }
        return metrics;
    }

    @Override
    public void trainEpochs(List<Pair<List<ObjectML>, Boolean>> dataset, int epochs) {
        if (dataset == null || dataset.isEmpty() || epochs <= 0) return;

        int bs;
        double labelSmoothing;
        readLock.lock();
        try {
            bs = Math.max(1, batchSize);
            labelSmoothing = cfg.labelSmoothing;
        } finally {
            readLock.unlock();
        }

        List<Pair<List<ObjectML>, Boolean>> shuffled = new ArrayList<>(dataset);
        java.util.Collections.shuffle(shuffled, rng);

        int total = shuffled.size();
        int splitIndex;
        if (total == 1) {
            splitIndex = 1;
        } else {
            splitIndex = (int) Math.floor(total * 0.8);
            splitIndex = Math.max(1, Math.min(splitIndex, total - 1));
        }

        List<Pair<List<ObjectML>, Boolean>> trainSet = new ArrayList<>(shuffled.subList(0, splitIndex));
        List<Pair<List<ObjectML>, Boolean>> validSet = splitIndex < total
                ? new ArrayList<>(shuffled.subList(splitIndex, total))
                : Collections.emptyList();

        Logger.info("Dataset split: " + trainSet.size() + " training samples, " + validSet.size() + " validation samples.");

        for (int e = 0; e < epochs; e++) {
            java.util.Collections.shuffle(trainSet, rng);
            List<Sample> currentBatch = new ArrayList<>(bs);
            int trainSkippedInvalid = 0;
            int trainSkippedNonFiniteProb = 0;

            for (Pair<List<ObjectML>, Boolean> dataPair : trainSet) {
                double y = dataPair.getY() ? 1.0 : 0.0;
                if (labelSmoothing > 0.0) {
                    y = y * (1.0 - labelSmoothing) + 0.5 * labelSmoothing;
                }

                double[][] vecs = prepareVectors(dataPair.getX());
                if (vecs.length < 2) continue;

                List<Sample> chunks = new ArrayList<>();
                appendChunks(vecs, y, chunks);
                for (Sample chunk : chunks) {
                    currentBatch.add(chunk);
                    if (currentBatch.size() >= bs) {
                        TrainBatchResult res;
                        writeLock.lock();
                        try {
                            res = trainBatch(currentBatch);
                        } finally {
                            writeLock.unlock();
                        }
                        trainSkippedInvalid += res.skippedInvalidSequence;
                        trainSkippedNonFiniteProb += res.skippedNonFiniteProb;
                        currentBatch.clear();
                    }
                }
            }
            if (!currentBatch.isEmpty()) {
                TrainBatchResult res;
                writeLock.lock();
                try {
                    res = trainBatch(currentBatch);
                } finally {
                    writeLock.unlock();
                }
                trainSkippedInvalid += res.skippedInvalidSequence;
                trainSkippedNonFiniteProb += res.skippedNonFiniteProb;
                currentBatch.clear();
            }

            DatasetMetrics trainMetrics;
            DatasetMetrics validMetrics;
            readLock.lock();
            try {
                trainMetrics = evaluateDataset(trainSet);
                validMetrics = evaluateDataset(validSet);
            } finally {
                readLock.unlock();
            }

            double avgTrainLoss = trainMetrics.avgLoss;
            double avgTrainAcc = trainMetrics.acc;
            double avgValidLoss = validMetrics.avgLoss;
            double acc = validMetrics.acc;
            double precision = (validMetrics.tp + validMetrics.fp) == 0 ? 0.0 : (double) validMetrics.tp / (validMetrics.tp + validMetrics.fp);
            double recall = (validMetrics.tp + validMetrics.fn) == 0 ? 0.0 : (double) validMetrics.tp / (validMetrics.tp + validMetrics.fn);
            double f1 = precision + recall == 0 ? 0.0 : 2 * precision * recall / (precision + recall);
            double fpr = (validMetrics.fp + validMetrics.tn) == 0 ? 0.0 : (double) validMetrics.fp / (validMetrics.fp + validMetrics.tn);

            double rocAuc = rocAuc(validMetrics.pairs);
            double prAuc = prAuc(validMetrics.pairs);

            Logger.info(String.format("Epoch %d/%d | Train [Loss: %.4f, Acc: %.1f%%] | Valid [Loss: %.4f, Acc: %.1f%%]",
                            (e + 1), epochs, avgTrainLoss, avgTrainAcc, avgValidLoss, acc));
            if (trainSkippedInvalid > 0
                    || trainSkippedNonFiniteProb > 0
                    || trainMetrics.skippedInvalidOrNonFinite > 0
                    || validMetrics.skippedInvalidOrNonFinite > 0) {
                Logger.warn(String.format(
                                "Skipped samples -> train chunks invalid: %d, train chunks non-finite probability: %d, train samples invalid/non-finite (eval): %d, validation samples invalid/non-finite (eval): %d",
                                trainSkippedInvalid, trainSkippedNonFiniteProb, trainMetrics.skippedInvalidOrNonFinite, validMetrics.skippedInvalidOrNonFinite));
            }
            Logger.info(String.format("Validation Metrics -> Precision: %.4f | Recall: %.4f | F1: %.4f | FPR: %.4f",
                            precision, recall, f1, fpr));
            Logger.info(String.format("Advanced Metrics -> ROC-AUC: %.4f | PR-AUC: %.4f", rocAuc, prAuc));
            Logger.info(String.format("Confusion Matrix -> TP: %d | TN: %d | FP: %d | FN: %d", validMetrics.tp, validMetrics.tn, validMetrics.fp, validMetrics.fn));
        }
    }
}
