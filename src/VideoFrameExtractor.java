import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

import java.io.File;

public class VideoFrameExtractor {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Define both folders that may contain videos
        String[] folders = { "data/idan", "data/unknown" };

        for (String folderPath : folders) {
            File folder = new File(folderPath);
            File[] files = folder.listFiles();

            if (files == null) {
                System.out.println("Could not access folder: " + folderPath);
                continue;
            }

            for (File file : files) {
                String filename = file.getName().toLowerCase();

                // Check if it's a video file (add more extensions if needed)
                if (filename.endsWith(".mov") || filename.endsWith(".mp4") || filename.endsWith(".avi")) {
                    String fullPath = file.getAbsolutePath();
                    System.out.println("Processing video: " + fullPath);

                    // Create VideoCapture instance
                    VideoCapture video = new VideoCapture(fullPath);

                    if (!video.isOpened()) {
                        System.out.println("Could not open video: " + fullPath);
                        continue;
                    }

                    Mat frame = new Mat();
                    int frameNumber = 0;

                    while (video.read(frame)) {
                        if (frameNumber % 5 == 0) { // Save every 5th frame
                            String frameName = file.getName().replaceAll("\\..*$", ""); // remove file extension
                            String outputFilename = folderPath + "/frame_" + frameName + "_" + frameNumber + ".jpg";
                            Imgcodecs.imwrite(outputFilename, frame);
                            System.out.println("Saved: " + outputFilename);
                        }
                        frameNumber++;
                    }

                    video.release();
                    System.out.println(" Finished extracting frames from: " + file.getName());
                }
            }
        }

        System.out.println(" All video processing complete!");
    }
}
