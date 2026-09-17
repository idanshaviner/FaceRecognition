package com.idanshaviner.facerecognition;

import org.opencv.core.*;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.face.FaceRecognizer;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Trains an LBPH face recognizer from images under data/idan (label 0) and
 * data/unknown (label 1), and saves it to models/model.xml.
 *
 * Run from the project root. data/ is gitignored — populate it yourself,
 * e.g. via VideoFrameExtractor, before running this.
 */
public class Trainer {
    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";
    static String modelFile = "models/model.xml";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        // Label 0 = you ("idan"), label 1 = unknown people
        loadImages("data/idan", 0, images, labels);
        loadImages("data/unknown", 1, images, labels);

        if (images.isEmpty()) {
            System.out.println("No training images found under data/idan or data/unknown - nothing to train.");
            return;
        }

        System.out.println("Training with " + images.size() + " face samples...");

        FaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.train(images, Converters.vector_int_to_Mat(labels));
        recognizer.save(modelFile);

        System.out.println("Training complete. Model saved as " + modelFile);
    }

    /**
     * Detects and crops the first face in every image in a folder, using the
     * same crop+resize as Camera.java's live recognition (see FaceUtils) so
     * training and inference stay consistent.
     */
    private static void loadImages(String folder, int label, List<Mat> images, List<Integer> labels) {
        File dir = new File(folder);
        File[] files = dir.listFiles();

        if (files == null || files.length == 0) {
            System.out.println("No files found in " + folder);
            return;
        }

        CascadeClassifier faceDetector = new CascadeClassifier(xmlFile);

        for (File file : files) {
            String filename = file.getName().toLowerCase();
            if (!filename.endsWith(".jpg") && !filename.endsWith(".jpeg") && !filename.endsWith(".png")) {
                System.out.println("Skipping non-image file: " + file.getName());
                continue;
            }

            Mat img = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            if (img.empty()) {
                System.out.println("Could not read image: " + file.getAbsolutePath());
                continue;
            }

            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(img, faceDetections);

            if (faceDetections.empty()) {
                System.out.println("No face detected in: " + file.getName());
                continue;
            }

            // use first detected face
            Rect face = faceDetections.toArray()[0];
            Mat faceROI = FaceUtils.extractFaceROI(img, face);

            images.add(faceROI);
            labels.add(label);

            System.out.println("Loaded " + file.getName() + " with label " + label);
        }
    }
}
