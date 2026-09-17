package com.idanshaviner.facerecognition;

import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

/**
 * Live webcam loop: detect faces with a Haar cascade, run each one through
 * the trained LBPH recognizer, and label it "idan" or "unknown".
 *
 * Run from the project root so the relative model paths below resolve.
 */
public class Camera {
    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";
    static String modelFile = "models/model.xml";

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("error opening camera");
            return;
        }

        // max resolution for better recognition
        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 1920);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 1080);

        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        FaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.read(modelFile);

        // ideal dimensions for face alignment (for user guidance)
        int idealWidth = 250;
        int idealHeight = 300;
        int toleranceWidth = 75;
        int toleranceHeight = 75;

        Mat frame = new Mat();
        while (camera.read(frame)) {
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            MatOfRect faceDetections = new MatOfRect();
            classifier.detectMultiScale(gray, faceDetections);

            // define safe zone center and bounds
            int ideal_X_pos = frame.cols() / 2;
            int ideal_Y_pos = (int) (frame.rows() * 0.35);
            int toleranceX = 300;
            int toleranceY = 350;
            int safePointX1 = ideal_X_pos - toleranceX / 2;
            int safePointX2 = ideal_X_pos + toleranceX / 2;
            int safePointY1 = ideal_Y_pos - toleranceY / 2;
            int safePointY2 = ideal_Y_pos + toleranceY / 2;

            // draw green box to show alignment zone
            Imgproc.rectangle(frame,
                    new Point(safePointX1, safePointY1),
                    new Point(safePointX2, safePointY2),
                    new Scalar(0, 255, 0), 2);

            for (Rect faceRect : faceDetections.toArray()) {
                // skip small faces that are unreliable
                if (faceRect.width < 100 || faceRect.height < 100) {
                    continue;
                }

                Imgproc.rectangle(frame,
                        new Point(faceRect.x, faceRect.y),
                        new Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height),
                        new Scalar(0, 0, 255), 3);

                int centerX = faceRect.x + faceRect.width / 2;
                int centerY = faceRect.y + faceRect.height / 2;

                String instructions = "";
                if (centerX < safePointX1) instructions = "move leftwards";
                else if (centerX > safePointX2) instructions = "move rightwards";
                else if (centerY < safePointY1) instructions = "move downwards";
                else if (centerY > safePointY2) instructions = "move upwards";

                String zInstructions = "";
                if (faceRect.width < idealWidth - toleranceWidth || faceRect.height < idealHeight - toleranceHeight)
                    zInstructions = "move closer to the screen";
                else if (faceRect.width > idealWidth + toleranceWidth || faceRect.height > idealHeight + toleranceHeight)
                    zInstructions = "move farther to the screen";

                // same crop+resize as training (FaceUtils) so LBPH compares like with like
                Mat faceROI = FaceUtils.extractFaceROI(gray, faceRect);

                int[] label = new int[1];
                double[] confidence = new double[1];
                recognizer.predict(faceROI, label, confidence);

                // labeling only idan or unknown
                double strictThreshold = 82;
                String name;
                if (label[0] == 0 && confidence[0] < strictThreshold) {
                    name = "idan";
                } else {
                    name = "unknown";
                }

                System.out.printf("predicted label: %d, confidence: %.2f, name: %s%n", label[0], confidence[0], name);
                System.out.println("face size: " + faceRect.width + "x" + faceRect.height);

                int textX = faceRect.x;
                Imgproc.putText(frame, name,
                        new Point(textX, faceRect.y - 40),
                        Imgproc.FONT_HERSHEY_TRIPLEX, 0.7,
                        new Scalar(255, 255, 0), 2);

                Imgproc.putText(frame, instructions.trim(),
                        new Point(textX, faceRect.y - 20),
                        Imgproc.FONT_HERSHEY_TRIPLEX, 0.7,
                        new Scalar(255, 255, 255), 2);

                Imgproc.putText(frame, zInstructions.trim(),
                        new Point(textX, faceRect.y),
                        Imgproc.FONT_HERSHEY_TRIPLEX, 0.7,
                        new Scalar(255, 255, 255), 2);
            }

            HighGui.imshow("FaceRecognition", frame);
            gray.release();

            // exit if esc is pressed
            if (HighGui.waitKey(2) == 27) break;
        }

        camera.release();
        HighGui.destroyAllWindows();
    }
}
