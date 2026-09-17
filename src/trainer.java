import org.opencv.core.*;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.face.FaceRecognizer;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class trainer {
    public static void main(String[] args) {

        // openCV native library
        System.loadLibrary("opencv_java4100");

        //store face images and their corresponding labels
        List<Mat> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        /*
         Load face data.
         - Label 0 = you ("idan")
         - Label 1 = unknown people
         - Each image will be cropped to only the face and resized for consistency
        */
        loadImages("data/idan", 0, images, labels);
        loadImages("data/unknown", 1, images, labels);

        System.out.println("Training with " + images.size() + " face samples...");

        FaceRecognizer recognizer = LBPHFaceRecognizer.create();

        // train recognizer with face images and their labels
        recognizer.train(images, Converters.vector_int_to_Mat(labels));

        // save the trained model
        recognizer.save("models/model.xml");

        System.out.println("Training complete. Model saved as models/model.xml");
    }

    /*
     Helper method to:
     - iterate through all files in a folder
     - skip non-image files
     - use Haar cascade to detect faces and crop
     - convert to grayscale and resize
     - add processed faces to the training set
    */
    private static void loadImages(String folder, int label, List<Mat> images, List<Integer> labels) {
        File dir = new File(folder);
        File[] files = dir.listFiles();

        if (files == null || files.length == 0) {
            System.out.println("No files found in " + folder);
            return;
        }

        // use same cascade file as camera class
        CascadeClassifier faceDetector = new CascadeClassifier("models/haarcascade_frontalface_alt2.xml");

        for (File file : files) {
            String filename = file.getName().toLowerCase();
            if (!filename.endsWith(".jpg") && !filename.endsWith(".jpeg") && !filename.endsWith(".png")) {
                System.out.println("Skipping non-image file: " + file.getName());
                continue;
            }

            // read as grayscale
            Mat img = Imgcodecs.imread(file.getAbsolutePath(), Imgcodecs.IMREAD_GRAYSCALE);
            if (img.empty()) {
                System.out.println("Could not read image: " + file.getAbsolutePath());
                continue;
            }

            // detect faces
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(img, faceDetections);

            if (faceDetections.empty()) {
                System.out.println("No face detected in: " + file.getName());
                continue;
            }

            // use first detected face
            Rect face = faceDetections.toArray()[0];
            Mat faceROI = new Mat(img, face);

            // resize face to standard size (200x200)
            Imgproc.resize(faceROI, faceROI, new Size(200, 200));

            // store image and label
            images.add(faceROI);
            labels.add(label);

            System.out.println("Loaded " + file.getName() + " with label " + label);
        }
    }
}
