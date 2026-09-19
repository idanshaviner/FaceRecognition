package com.idanshaviner.facerecognition;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Loads labelled face crops from a folder. Shared by Trainer and Evaluator so
 * both preprocess images identically to each other and to Camera.
 */
public final class FaceDataset {

    public static final int LABEL_IDAN = 0;
    public static final int LABEL_UNKNOWN = 1;

    /**
     * group identifies images that are not independent of each other (frames
     * cut from the same video), so Evaluator can keep them on one side of a
     * train/test split.
     */
    public record Sample(Mat face, int label, String group, String fileName) {
    }

    // VideoFrameExtractor writes frame_<videoName>_<frameNumber>.jpg
    private static final Pattern EXTRACTED_FRAME = Pattern.compile("^frame_(.+)_\\d+\\.[^.]+$");

    private FaceDataset() {
    }

    /**
     * @param detector Haar cascade used to find the face in each image, or null
     *                 if the images are already tightly cropped faces
     * @param verbose  print a line per loaded image (skips are always printed)
     */
    public static List<Sample> load(String folder, int label, CascadeClassifier detector, boolean verbose) {
        List<Sample> samples = new ArrayList<>();

        File[] files = new File(folder).listFiles(f -> f.isFile() && !f.getName().startsWith("."));
        if (files == null || files.length == 0) {
            System.out.println("No files found in " + folder);
            return samples;
        }
        Arrays.sort(files);

        if (detector != null && detector.empty()) {
            throw new IllegalStateException("Could not load the Haar cascade - run from the project root");
        }

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

            Rect faceRect;
            if (detector == null) {
                faceRect = new Rect(0, 0, img.cols(), img.rows());
            } else {
                MatOfRect faceDetections = new MatOfRect();
                detector.detectMultiScale(img, faceDetections);
                if (faceDetections.empty()) {
                    System.out.println("No face detected in: " + file.getName());
                    continue;
                }
                // use first detected face
                faceRect = faceDetections.toArray()[0];
            }

            samples.add(new Sample(FaceUtils.extractFaceROI(img, faceRect), label, groupOf(file.getName()), file.getName()));

            if (verbose) {
                System.out.println("Loaded " + file.getName() + " with label " + label);
            }
        }
        return samples;
    }

    static String groupOf(String fileName) {
        Matcher m = EXTRACTED_FRAME.matcher(fileName);
        return m.matches() ? "video:" + m.group(1) : fileName;
    }
}
