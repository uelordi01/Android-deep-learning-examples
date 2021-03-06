package org.uelordi.deepsamples.dlib;
import android.util.Log;

/**
 * Created by uelordi on 6/10/17.
 */

public class JniManager {
    static String ext="";

    static String[] libraryList_withPrefix = {
            "opencv_java3",
            "native-lib"
    };

    private static final String LOG_TAG = "JniManager";

    static {
        for (String aLibraryList_withPrefix : libraryList_withPrefix) {
            try {
                Log.v(LOG_TAG, "loading library -> " + aLibraryList_withPrefix + ext);
                System.loadLibrary(aLibraryList_withPrefix + ext);
            } catch (UnsatisfiedLinkError e) {
                System.err.println("Public native code library '" + aLibraryList_withPrefix + "' failed to load.\n" + e);
                Log.e(LOG_TAG, e.getMessage());
            }
        }

    }

    /**
     * @param deepNetFilename: the neural protobuf definet net (JSON)
     * @param weightsFilename: the weights file that is trained
     */
    public static native void init(String deepNetFilename, String weightsFilename);

    /**
     * \Brief processes the image through the neural net.
     * @param colorImage:
     * @param greyImage
     */
    public static native void process(long colorImage, long greyImage);

    /**
     * \Brief make the call to start processing
     */
    public static native void start();

    /**
     * \Brief make stop function to stop processing:
     */
    public static native void stop();

}
