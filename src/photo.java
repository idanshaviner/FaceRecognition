
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



public class photo {



    static String imageFile="/Users/idanshaviner/nito.jpeg";
    static String xmlFile = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";

    //	static String xmlFile = "lbpcascade_frontalface.xml";

    public static void main(String[] args) {

        //openCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // open image file

        VideoCapture image = new VideoCapture(imageFile);


        if (!image.isOpened()) {
            System.out.println("Error opening image file");
            return;
        }

        CascadeClassifier classifier = new CascadeClassifier(xmlFile);


        Mat frame = new Mat();
        while (image.read(frame)) {

            HighGui.imshow("Face Detection", frame);


            MatOfRect faceDetections = new MatOfRect();
            classifier.detectMultiScale(frame, faceDetections);
            //System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frame,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 0, 255), 3
                );
            }

            HighGui.waitKey(0);

        }

        image.release();
        HighGui.destroyAllWindows();
    }
}