package org.uelordi.deepsamples.caffe2;

import android.content.res.AssetManager;
import android.util.Log;

/**
 * Created by uelordi on 6/10/17.
 */

public class JniManager {
    static String ext="";

    static String[] libraryList_withPrefix = {
//            "opencv_java3",
            "glog",
            "gnustl_shared",
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
    public static native void init(AssetManager mgr);
    public static native void start();
    public static native void stop();
    public static native String processYUVFrame(int h, int w, byte[] Y, byte[] U, byte[] V,
                                                  int rowStride, int pixelStride, boolean r_hwc);

}
