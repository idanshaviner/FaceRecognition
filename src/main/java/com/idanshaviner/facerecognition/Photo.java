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
 * Detection-only smoke test against a single image: draws a box around every
 * face found, no recognition. Useful for sanity-checking the cascade file
 * without needing a trained model or a webcam.
 *
 * Usage: run from the project root with the image path as the first arg,
 *   e.g. `mvn exec:java -Dexec.mainClass=... -Dexec.args=/path/to/photo.jpg`
 */
public class Photo {
    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: Photo <path-to-image>");
            return;
        }
        String imageFile = args[0];

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        VideoCapture image = new VideoCapture(imageFile);
        if (!image.isOpened()) {
            System.out.println("Error opening image file: " + imageFile);
            return;
        }

        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        Mat frame = new Mat();
        while (image.read(frame)) {
            MatOfRect faceDetections = new MatOfRect();
            classifier.detectMultiScale(frame, faceDetections);

            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frame,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 0, 255), 3
                );
            }

            HighGui.imshow("Face Detection", frame);
            HighGui.waitKey(0);
        }

        image.release();
        HighGui.destroyAllWindows();
    }
}
