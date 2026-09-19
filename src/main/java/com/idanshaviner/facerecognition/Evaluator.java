package com.idanshaviner.facerecognition;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;

/**
 * Measures how well the recognizer works, using the same preprocessing
 * (FaceDataset / FaceUtils) and the same accept rule (FaceUtils.isIdan) as
 * Trainer and Camera, so the numbers describe what actually runs live.
 *
 * Method: grouped, stratified k-fold cross-validation over data/idan and
 * data/unknown. Each fold trains a fresh model on the other folds and predicts
 * the held-out faces. Frames extracted from the same video are kept in the
 * same fold - otherwise near-identical frames land on both sides of the split
 * and the results look far better than they are.
 *
 * Reported per threshold:
 *   FAR (false accept rate) = unknown faces wrongly labelled "idan"
 *   FRR (false reject rate) = your faces wrongly labelled "unknown"
 *
 * Usage (from the project root):
 *   Evaluator [--folds N] [--seed S] [--far-target F] [--no-detect]
 * --no-detect treats every image as an already-cropped face (skips Haar).
 */
public class Evaluator {

    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";

    record Result(String fileName, int trueLabel, int predictedLabel, double distance) {
    }

    record Rates(int falseAccepts, int impostors, int falseRejects, int genuine) {
        double far() {
            return impostors == 0 ? 0 : (double) falseAccepts / impostors;
        }

        double frr() {
            return genuine == 0 ? 0 : (double) falseRejects / genuine;
        }
    }

    record Operating(double threshold, Rates rates) {
    }

    public static void main(String[] args) {
        int folds = 5;
        long seed = 42;
        double farTarget = 0.01;
        boolean detect = true;

        try {
            for (int i = 0; i < args.length; i++) {
                switch (args[i]) {
                    case "--folds" -> folds = Integer.parseInt(args[++i]);
                    case "--seed" -> seed = Long.parseLong(args[++i]);
                    case "--far-target" -> farTarget = Double.parseDouble(args[++i]);
                    case "--no-detect" -> detect = false;
                    default -> throw new IllegalArgumentException("Unknown argument: " + args[i]);
                }
            }
        } catch (RuntimeException e) {
            System.out.println(e.getMessage() == null ? "Bad arguments" : e.getMessage());
            System.out.println("Usage: Evaluator [--folds N] [--seed S] [--far-target F] [--no-detect]");
            return;
        }
        if (folds < 2) {
            System.out.println("--folds must be at least 2");
            return;
        }

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        CascadeClassifier detector = detect ? new CascadeClassifier(xmlFile) : null;
        List<FaceDataset.Sample> idan = FaceDataset.load("data/idan", FaceDataset.LABEL_IDAN, detector, false);
        List<FaceDataset.Sample> unknown = FaceDataset.load("data/unknown", FaceDataset.LABEL_UNKNOWN, detector, false);

        Map<String, List<FaceDataset.Sample>> idanGroups = groupSamples(idan);
        Map<String, List<FaceDataset.Sample>> unknownGroups = groupSamples(unknown);

        System.out.println();
        System.out.printf("Loaded %d idan faces (%d independent groups), %d unknown faces (%d independent groups)%n",
                idan.size(), idanGroups.size(), unknown.size(), unknownGroups.size());

        int maxFolds = Math.min(idanGroups.size(), unknownGroups.size());
        if (maxFolds < 2) {
            System.out.println("Need at least 2 independent groups (separate photos or separate videos) in BOTH");
            System.out.println("data/idan and data/unknown to hold any data out. Add more and re-run.");
            return;
        }
        if (folds > maxFolds) {
            System.out.printf("Only %d independent groups available in the smaller class - using %d folds instead of %d%n",
                    maxFolds, maxFolds, folds);
            folds = maxFolds;
        }

        Random rnd = new Random(seed);
        Map<String, Integer> foldOfGroup = new LinkedHashMap<>();
        assignFolds(idanGroups, folds, rnd, foldOfGroup);
        assignFolds(unknownGroups, folds, rnd, foldOfGroup);

        List<FaceDataset.Sample> all = new ArrayList<>(idan);
        all.addAll(unknown);

        List<Result> results = new ArrayList<>();
        System.out.printf("Running %d-fold cross-validation (seed %d)...%n", folds, seed);
        for (int f = 0; f < folds; f++) {
            List<Mat> trainImages = new ArrayList<>();
            List<Integer> trainLabels = new ArrayList<>();
            List<FaceDataset.Sample> test = new ArrayList<>();
            for (FaceDataset.Sample s : all) {
                if (foldOfGroup.get(key(s)) == f) {
                    test.add(s);
                } else {
                    trainImages.add(s.face());
                    trainLabels.add(s.label());
                }
            }

            FaceRecognizer recognizer = LBPHFaceRecognizer.create();
            recognizer.train(trainImages, Converters.vector_int_to_Mat(trainLabels));

            for (FaceDataset.Sample s : test) {
                int[] label = new int[1];
                double[] distance = new double[1];
                recognizer.predict(s.face(), label, distance);
                results.add(new Result(s.fileName(), s.label(), label[0], distance[0]));
            }
            System.out.printf("  fold %d: trained on %d, tested on %d%n", f + 1, trainImages.size(), test.size());
        }

        report(results, farTarget);
    }

    static Map<String, List<FaceDataset.Sample>> groupSamples(List<FaceDataset.Sample> samples) {
        Map<String, List<FaceDataset.Sample>> groups = new LinkedHashMap<>();
        for (FaceDataset.Sample s : samples) {
            groups.computeIfAbsent(s.group(), g -> new ArrayList<>()).add(s);
        }
        return groups;
    }

    private static String key(FaceDataset.Sample s) {
        return s.label() + "|" + s.group();
    }

    /** Shuffle a class's groups, then deal them round-robin so every fold gets some. */
    private static void assignFolds(Map<String, List<FaceDataset.Sample>> groups, int folds, Random rnd,
                                    Map<String, Integer> foldOfGroup) {
        List<String> names = new ArrayList<>(groups.keySet());
        Collections.sort(names);
        Collections.shuffle(names, rnd);
        for (int i = 0; i < names.size(); i++) {
            FaceDataset.Sample any = groups.get(names.get(i)).get(0);
            foldOfGroup.put(key(any), i % folds);
        }
    }

    static Rates ratesAt(List<Result> results, double threshold) {
        int falseAccepts = 0, impostors = 0, falseRejects = 0, genuine = 0;
        for (Result r : results) {
            boolean accepted = FaceUtils.isIdan(r.predictedLabel(), r.distance(), threshold);
            if (r.trueLabel() == FaceDataset.LABEL_IDAN) {
                genuine++;
                if (!accepted) falseRejects++;
            } else {
                impostors++;
                if (accepted) falseAccepts++;
            }
        }
        return new Rates(falseAccepts, impostors, falseRejects, genuine);
    }

    /** 0, 0.5, 1.0, ... just past the largest distance the model gave for label 0. */
    static double[] candidateThresholds(List<Result> results) {
        double max = 0;
        for (Result r : results) {
            if (r.predictedLabel() == FaceDataset.LABEL_IDAN && Double.isFinite(r.distance())) {
                max = Math.max(max, r.distance());
            }
        }
        int n = (int) Math.ceil(max * 2) + 2;
        double[] t = new double[n];
        for (int i = 0; i < n; i++) t[i] = i * 0.5;
        return t;
    }

    /** Threshold where FAR and FRR are closest; ties go to the stricter (lower) threshold. */
    static Operating equalErrorPoint(List<Result> results, double[] thresholds) {
        Operating best = null;
        double bestGap = Double.MAX_VALUE;
        for (double t : thresholds) {
            Rates r = ratesAt(results, t);
            double gap = Math.abs(r.far() - r.frr());
            if (gap < bestGap - 1e-12) {
                bestGap = gap;
                best = new Operating(t, r);
            }
        }
        return best;
    }

    /** Lowest FRR with FAR at or under the target; ties go to the stricter (lower) threshold. */
    static Operating bestAtFar(List<Result> results, double[] thresholds, double farTarget) {
        Operating best = null;
        for (double t : thresholds) {
            Rates r = ratesAt(results, t);
            if (r.far() <= farTarget + 1e-12 && (best == null || r.frr() < best.rates().frr() - 1e-12)) {
                best = new Operating(t, r);
            }
        }
        return best;
    }

    /** The chosen threshold is the lowest of a run of thresholds that give identical results; show how far it runs. */
    static String sameResultUntil(List<Result> results, double[] thresholds, Operating op) {
        double last = op.threshold();
        for (double t : thresholds) {
            if (t < op.threshold()) continue;
            Rates r = ratesAt(results, t);
            if (r.falseAccepts() != op.rates().falseAccepts() || r.falseRejects() != op.rates().falseRejects()) break;
            last = t;
        }
        if (last == op.threshold()) return "";
        return last == thresholds[thresholds.length - 1]
                ? String.format("  (same result from %.1f upward)", op.threshold())
                : String.format("  (same result from %.1f to %.1f)", op.threshold(), last);
    }

    private static void report(List<Result> results, double farTarget) {
        double[] thresholds = candidateThresholds(results);
        Rates genuineCounts = ratesAt(results, 0);

        System.out.println();
        System.out.println("=== Results (each face scored by a model that never saw it, or any frame of the same video) ===");
        if (genuineCounts.genuine() < 30 || genuineCounts.impostors() < 30) {
            System.out.printf("Note: only %d idan and %d unknown test faces - rates are coarse; one face is %.1f%% / %.1f%%.%n",
                    genuineCounts.genuine(), genuineCounts.impostors(),
                    100.0 / Math.max(1, genuineCounts.genuine()), 100.0 / Math.max(1, genuineCounts.impostors()));
        }

        System.out.println();
        System.out.println("Threshold sweep (accept as idan only if label==0 AND distance < threshold):");
        TreeSet<Double> rows = new TreeSet<>();
        for (int t = 20; t <= 200; t += 20) rows.add((double) t);
        rows.add(Camera.STRICT_THRESHOLD);
        for (double t : rows) {
            System.out.println(line(String.format("threshold %5.1f", t), ratesAt(results, t),
                    t == Camera.STRICT_THRESHOLD ? "  <-- current Camera threshold" : ""));
        }
        System.out.println(line("label only (no cutoff)", ratesAt(results, Double.POSITIVE_INFINITY), ""));

        Operating current = new Operating(Camera.STRICT_THRESHOLD, ratesAt(results, Camera.STRICT_THRESHOLD));
        Operating eer = equalErrorPoint(results, thresholds);
        Operating atTarget = bestAtFar(results, thresholds, farTarget);

        System.out.println();
        System.out.println("Operating points:");
        System.out.println(line(String.format("current  (%.1f)", current.threshold()), current.rates(), ""));
        boolean crosses = Math.abs(eer.rates().far() - eer.rates().frr()) <= 0.05;
        System.out.println(line(String.format("equal-error (%.1f)", eer.threshold()), eer.rates(),
                crosses
                        ? String.format("  EER ~ %.1f%%%s", 50 * (eer.rates().far() + eer.rates().frr()),
                        sameResultUntil(results, thresholds, eer))
                        : "  (FAR and FRR never cross - the threshold can't fix this, the classifier itself is failing)"));
        System.out.println(line(String.format("FAR <= %.1f%% (%.1f)", 100 * farTarget, atTarget.threshold()),
                atTarget.rates(), sameResultUntil(results, thresholds, atTarget)));

        printHardCases(results);

        System.out.println();
        System.out.println("Reading this:");
        System.out.println(" - Lower threshold = stricter: FAR falls, FRR rises. Pick the trade-off you can live with.");
        System.out.println(" - If FAR is 0 even at high thresholds, the 'unknown' class alone is rejecting strangers on this");
        System.out.println("   data; that says little about faces unlike anyone in data/unknown.");
        System.out.println(" - Photos from one session (same lighting/pose) are near-duplicates but can't be grouped");
        System.out.println("   automatically, so results can still be optimistic. Use separate sessions when you can.");
    }

    private static String line(String label, Rates r, String suffix) {
        return String.format("  %-24s FAR %2d/%-3d (%5.1f%%)   FRR %2d/%-3d (%5.1f%%)%s",
                label, r.falseAccepts(), r.impostors(), 100 * r.far(), r.falseRejects(), r.genuine(), 100 * r.frr(), suffix);
    }

    private static void printHardCases(List<Result> results) {
        System.out.println();
        System.out.println("Unknown faces that came closest to being accepted as idan (label 0, smallest distance):");
        List<Result> impostors = new ArrayList<>();
        List<Result> genuine = new ArrayList<>();
        for (Result r : results) {
            if (r.trueLabel() == FaceDataset.LABEL_UNKNOWN && r.predictedLabel() == FaceDataset.LABEL_IDAN) impostors.add(r);
            if (r.trueLabel() == FaceDataset.LABEL_IDAN) genuine.add(r);
        }
        impostors.sort(Comparator.comparingDouble(Result::distance));
        if (impostors.isEmpty()) {
            System.out.println("  none - every unknown face was classified as label 1");
        }
        for (Result r : impostors.subList(0, Math.min(5, impostors.size()))) {
            System.out.printf("  %-40s distance %.1f%n", r.fileName(), r.distance());
        }

        System.out.println("Your faces that would be rejected first (predicted unknown, else largest distance):");
        genuine.sort(Comparator.comparingDouble((Result r) ->
                r.predictedLabel() == FaceDataset.LABEL_IDAN ? r.distance() : Double.POSITIVE_INFINITY).reversed());
        for (Result r : genuine.subList(0, Math.min(5, genuine.size()))) {
            System.out.printf("  %-40s predicted %d, distance %.1f%n", r.fileName(), r.predictedLabel(), r.distance());
        }
    }
}
