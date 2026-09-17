package com.idanshaviner.facerecognition;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

/**
 * Detection-only smoke test against a video file: draws a box around every
 * face found per frame, no recognition.
 *
 * An early version of this class rendered frames by hand (Mat -> JPEG bytes
 * -> BufferedImage -> a Swing JLabel) before this project settled on
 * OpenCV's own HighGui.imshow, which does the same job in one call.
 *
 * Usage: run from the project root with the video path as the first arg.
 */
public class Video {
    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: Video <path-to-video>");
            return;
        }
        String videoFile = args[0];

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        VideoCapture video = new VideoCapture(videoFile);
        if (!video.isOpened()) {
            System.out.println("Error opening video file: " + videoFile);
            return;
        }

        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        Mat frame = new Mat();
        while (video.read(frame)) {
            HighGui.imshow("Face Detection", frame);

            MatOfRect faceDetections = new MatOfRect();
            classifier.detectMultiScale(frame, faceDetections);

            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frame,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 0, 255), 3
                );
            }

            if (HighGui.waitKey(2) == 27) {
                break;
            }
        }

        video.release();
        HighGui.destroyAllWindows();
    }
}
