package org.uelordi.deepsamples.caffe2;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.TextureView;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.uelordi.deepsamples.caffe2.camera.CameraHandler;

import java.util.ArrayList;
import static android.view.View.SYSTEM_UI_FLAG_IMMERSIVE;
public class MainActivity extends AppCompatActivity implements org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2 {

    // Used to load the 'native-lib' library on application startup.
    private CameraBridgeViewBase mOpenCvCameraView;
    private static final String TAG = "MainActivity";
    private Mat mRgba;
    private Mat mGray;
    TextView mresult_text;
    TextureView mCameraView;
    private String predictedClass = "none";
    private int[] IMAGE_RESOLUTION = {320, 240};

    enum CAMERA_TYPE {OPENCV_CAPTURE, NATIVE_CAPTURE}

   Activity act ;
    CameraHandler m_camHandler;
    static CAMERA_TYPE mCaptureType = CAMERA_TYPE.NATIVE_CAPTURE;
    private BaseLoaderCallback mLoaderCallback;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        askPermissions();
        mresult_text = (TextView) (findViewById(R.id.result_text));
        mCameraView = (TextureView) findViewById(R.id.caffe2_camera_view);
        mCameraView.setSystemUiVisibility(SYSTEM_UI_FLAG_IMMERSIVE);
        act = this;
        new SetUpNeuralNetwork().execute();
    }

    @Override
    public void onCameraViewStarted(int i, int i1) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        JniManager.process(mRgba.getNativeObjAddr(), mGray.getNativeObjAddr());
        return mRgba;
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mCaptureType == CAMERA_TYPE.OPENCV_CAPTURE) {
            if (!OpenCVLoader.initDebug()) {
                Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
            } else {
                Log.d(TAG, "OpenCV library found inside package. Using it!");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            }
            mOpenCvCameraView.enableView();
        }

    }

    @Override
    protected void onPause() {
        super.onPause();
        JniManager.stop();
    }

    void askPermissions() {
        final String[] permissions_needed = {Manifest.permission.CAMERA,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
                //,Manifest.permission.SYSTEM_ALERT_WINDOW
        };
        final int REQUEST_CODE_ASK_PERMISSIONS = 123;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.M) {
            ArrayList array_aux = new ArrayList<String>();

            String[] permissions_requested;
            for (String aPermissions_needed : permissions_needed) {
                int hasPermission = ContextCompat.checkSelfPermission(this, aPermissions_needed);
                if (hasPermission != PackageManager.PERMISSION_GRANTED) {
                    array_aux.add(aPermissions_needed);
                    Log.d("Activity", "request permission %s" + aPermissions_needed);
                }
            }
            if (array_aux.size() != 0) {
                //Log.d("Activity", "allow permission");
                permissions_requested = (String[]) array_aux.toArray(new String[array_aux.size()]);


                ActivityCompat.requestPermissions(this, permissions_requested,
                        REQUEST_CODE_ASK_PERMISSIONS);
            }
        } else {
            Log.v(TAG, "No need to ask permissions programatically");
        }

    }

    private class SetUpNeuralNetwork extends AsyncTask<Void, Void, Void> {
        @Override
        protected Void doInBackground(Void[] v) {
            try {
                JniManager.init(getResources().getAssets());
                predictedClass = "Neural net loaded! Inferring...";
            } catch (Exception e) {
                Log.d(TAG, "Couldn't load neural network.");
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            switch (mCaptureType) {
                case NATIVE_CAPTURE:
                    m_camHandler = new CameraHandler();
                    m_camHandler.setCallback(new CameraHandler.CameraCallback() {
                        @Override
                        public void onGetFrame(byte[] Y, byte[] U, byte[] V, int rowStride, int pixelStride, int width, int height, long timestamp) {
                            predictedClass = JniManager.processYUVFrame(height, width, Y, U, V, rowStride, pixelStride, false);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    mresult_text.setText(predictedClass);
                                }
                            });
                        }

                        @Override
                        public void onConfiguration(int camId, int deviceOrientation, int sensorOrientation) {

                        }
                    });

                    m_camHandler.init(act, CameraHandler.CameraID.FRONT,
                            IMAGE_RESOLUTION[0],
                            IMAGE_RESOLUTION[1]);
                    m_camHandler.setTextureView(mCameraView);

                    m_camHandler.start();
                    JniManager.start();
                    break;
                case OPENCV_CAPTURE:
                    mLoaderCallback = new BaseLoaderCallback(act) {
                        @Override
                        public void onManagerConnected(int status) {
                            super.onManagerConnected(status);
                            switch (status) {
                                case LoaderCallbackInterface.SUCCESS:
                                    break;
                                case LoaderCallbackInterface.INIT_FAILED:
                                    break;
                            }
                            Log.i(TAG, "OpenCV loaded successfully");
                        }
                    };
                    mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_surface);
                    mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
                    mOpenCvCameraView.setMaxFrameSize(320, 240);
                    mOpenCvCameraView.setCvCameraViewListener((CameraBridgeViewBase.CvCameraViewListener) act);
                    break;
            }

        }
    }
}
