package com.idanshaviner.facerecognition;

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Shared face preprocessing so training and live recognition see images
 * prepared the exact same way. Trainer.java and Camera.java both call
 * {@link #extractFaceROI} — if they ever diverge again, LBPH ends up
 * comparing histograms computed from differently-framed crops, which
 * silently tanks recognition accuracy (train/inference skew).
 */
public final class FaceUtils {

    public static final int FACE_SIZE = 200;

    private FaceUtils() {
    }

    public static Mat extractFaceROI(Mat grayFrame, Rect faceRect) {
        Mat faceROI = new Mat(grayFrame, faceRect);
        Imgproc.resize(faceROI, faceROI, new Size(FACE_SIZE, FACE_SIZE));
        return faceROI;
    }
}
