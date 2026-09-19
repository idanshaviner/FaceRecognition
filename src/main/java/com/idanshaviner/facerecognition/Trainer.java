package com.idanshaviner.facerecognition;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.List;

/**
 * Trains an LBPH face recognizer from images under data/idan (label 0) and
 * data/unknown (label 1), and saves it to models/model.xml.
 *
 * Run from the project root. data/ is gitignored — populate it yourself,
 * e.g. via VideoFrameExtractor, before running this. Run Evaluator first to
 * measure how well this setup actually recognizes you.
 */
public class Trainer {
    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";
    static String modelFile = "models/model.xml";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        CascadeClassifier faceDetector = new CascadeClassifier(xmlFile);

        List<FaceDataset.Sample> samples = new ArrayList<>();
        samples.addAll(FaceDataset.load("data/idan", FaceDataset.LABEL_IDAN, faceDetector, true));
        samples.addAll(FaceDataset.load("data/unknown", FaceDataset.LABEL_UNKNOWN, faceDetector, true));

        if (samples.isEmpty()) {
            System.out.println("No training images found under data/idan or data/unknown - nothing to train.");
            return;
        }

        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (FaceDataset.Sample s : samples) {
            images.add(s.face());
            labels.add(s.label());
        }

        System.out.println("Training with " + images.size() + " face samples...");

        FaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.train(images, Converters.vector_int_to_Mat(labels));
        recognizer.save(modelFile);

        System.out.println("Training complete. Model saved as " + modelFile);
    }
}
