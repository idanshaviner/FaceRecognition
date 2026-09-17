
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;


public class video {

    static String videoFile="/Users/idanshaviner/Downloads/pets.mp4";
    static String xmlFile = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml";


    public static void main(String[] args) {

        // openCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // open video file
        VideoCapture video = new VideoCapture(videoFile);

        if (!video.isOpened()) {
            System.out.println("Error opening video file");
            return;
        }

        CascadeClassifier classifier = new CascadeClassifier(xmlFile);

        Mat frame = new Mat();
        while (video.read(frame)) {

            HighGui.imshow("Face Detection", frame);

            MatOfRect faceDetections = new MatOfRect();
            classifier.detectMultiScale(frame, faceDetections);
            //System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(frame,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height), // top right
                        new Scalar(0, 0, 255), 3 // RGB colour & line width
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
	   

	 /*
	public static void main(String[] args) {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        
		JFrame jFrame = new JFrame();
		JLabel jLabel = new JLabel();
		jFrame.setContentPane(jLabel);
		jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jFrame.setVisible(true);
		jFrame.setBounds(0, 0, 1280, 720);
		
		
		//video caputure setting
		
		////VideoCapture = cv2.VideoCapture(0); // initialize, # is camera number
		//capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280); //CAP_PROP_FRAME_WIDTH == 3
		//capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720); //CAP_PROP_FRAME_HEIGHT == 4
		
		
		//video = new VideoCapture(0);
		
		VideoCapture videoCapture = new VideoCapture();
		
			
		videoCapture.open(videoFile);
					
        CascadeClassifier classifier = new CascadeClassifier(xmlFile);	
		
		Mat frame = new Mat();

		try {
			
			//while (camera.read(frame)) {
			while (videoCapture.read(frame)) {
								
				MatOfRect faceDetections = new MatOfRect();
				classifier.detectMultiScale(frame, faceDetections);
				//System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

				for (Rect rect : faceDetections.toArray()) {
					Imgproc.rectangle(frame, // where to draw the box
							new Point(rect.x, rect.y), // bottom left
							new Point(rect.x + rect.width, rect.y + rect.height), // top right
							new Scalar(0, 255, 0), 3 // RGB colour & line width
					);
				}						
				
				ImageIcon image = new ImageIcon(Mat2BufferedImage(frame));
				jLabel.setIcon(image);
				jLabel.repaint();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	
	private static BufferedImage Mat2BufferedImage(Mat matrix) throws Exception {
		MatOfByte matOfByte = new MatOfByte();
		Imgcodecs.imencode(".jpg", matrix, matOfByte);
		byte ba[] = matOfByte.toArray();
		BufferedImage bi = ImageIO.read(new ByteArrayInputStream(ba));
		return bi;
	}
	
	*/
	 


