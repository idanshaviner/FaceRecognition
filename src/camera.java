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

public class camera {
    // haar cascade xml file for face detection
    static String xmlFile = "models/haarcascade_frontalface_alt2.xml";

    public static void main(String[] args) {
        // opencv native library
        System.loadLibrary("opencv_java4100");

        // open default camera
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("error opening camera");
            return;
        }

        // max resolution for better recognition
        camera.set(Videoio.CAP_PROP_FRAME_WIDTH, 1920);
        camera.set(Videoio.CAP_PROP_FRAME_HEIGHT, 1080);

        // load the haar cascade classifier for face detection
        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        // load trained lbph model for face recognition
        FaceRecognizer recognizer = LBPHFaceRecognizer.create();
        recognizer.read("models/model.xml");

        // ideal dimensions for face alignment (for user guidance)
        int idealWidth = 250;
        int idealHeight = 300;
        int toleranceWidth = 75;
        int toleranceHeight = 75;

        // loop to capture video frames
        Mat frame = new Mat();
        while (camera.read(frame)) {
            // convert frame to grayscale (better for processing)
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            // detect faces
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

                // draw red rectangle around detected face
                Imgproc.rectangle(frame,
                        new Point(faceRect.x, faceRect.y),
                        new Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height),
                        new Scalar(0, 0, 255), 3);

                // find face center position
                int CenterX = faceRect.x + faceRect.width / 2;
                int CenterY = faceRect.y + faceRect.height / 2;

                // instructions for moving left/right/up/down
                String instructions = "";
                if (CenterX < safePointX1) instructions = "move leftwards";
                else if (CenterX > safePointX2) instructions = "move rightwards";
                else if (CenterY < safePointY1) instructions = "move downwards";
                else if (CenterY > safePointY2) instructions = "move upwards";

                // instructions for moving closer/farther
                String zInstructions = "";
                if (faceRect.width < idealWidth - toleranceWidth || faceRect.height < idealHeight - toleranceHeight)
                    zInstructions = "move closer to the screen";
                else if (faceRect.width > idealWidth + toleranceWidth || faceRect.height > idealHeight + toleranceHeight)
                    zInstructions = "move farther to the screen";

                // crop inside the detected face rectangle to avoid hair
                int cropX = faceRect.x + faceRect.width / 8;
                int cropY = faceRect.y + faceRect.height / 6;
                int cropWidth = faceRect.width * 3 / 4;
                int cropHeight = faceRect.height * 2 / 3;

                // make sure the crop stays inside the frame
                cropX = Math.max(cropX, 0);
                cropY = Math.max(cropY, 0);
                cropWidth = Math.min(cropWidth, frame.cols() - cropX);
                cropHeight = Math.min(cropHeight, frame.rows() - cropY);

                // prepare cropped face roi
                Rect croppedFace = new Rect(cropX, cropY, cropWidth, cropHeight);
                Mat faceROI = new Mat(gray, croppedFace);
                Imgproc.resize(faceROI, faceROI, new Size(200, 200));

                // predict label and confidence
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

                // console debugging
                System.out.printf("predicted label: %d, confidence: %.2f, name: %s%n", label[0], confidence[0], name);
                System.out.println("face size: " + faceRect.width + "x" + faceRect.height);

                // show name above face
                int textX = faceRect.x;
                Imgproc.putText(frame, name,
                        new Point(textX, faceRect.y - 40),
                        Imgproc.FONT_HERSHEY_TRIPLEX, 0.7,
                        new Scalar(255, 255, 0), 2);

                // show position guidance
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

        // clean all
        camera.release();
        HighGui.destroyAllWindows();
    }
}
