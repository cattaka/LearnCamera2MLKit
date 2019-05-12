package com.example.android.camera2basic

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Continuation
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModel
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel
import com.google.firebase.ml.custom.*
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.text.FirebaseVisionText
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class TextRecoginizer {
    companion object {
        private val DIM_BATCH_SIZE = 1
        private val DIM_PIXEL_SIZE = 3
        private val DIM_IMG_SIZE_X = 224
        private val DIM_IMG_SIZE_Y = 224
        private val intValues = IntArray(DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y)

        private val HOSTED_MODEL_NAME = "cloud_model_1"
        private val LOCAL_MODEL_ASSET = "mobilenet_v1_1.0_224_quant.tflite"
        private val LABEL_PATH = "labels.txt"

        private val RESULTS_TO_SHOW = 3
        private val sortedLabels = PriorityQueue<Map.Entry<String, Float>>(
                RESULTS_TO_SHOW,
                Comparator<Map.Entry<String, Float>> { o1, o2 -> o1.value.compareTo(o2.value) })

    }

    private lateinit var mLabelList: List<String>
    private lateinit var mDataOptions: FirebaseModelInputOutputOptions
    private var mInterpreter: FirebaseModelInterpreter? = null

    fun initCustomModel(context: Context) {
        mLabelList = loadLabelList(context)

        val inputDims = intArrayOf(DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE)
        val outputDims = intArrayOf(DIM_BATCH_SIZE, mLabelList.size)

        mDataOptions = FirebaseModelInputOutputOptions.Builder()
                .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
                .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
                .build()
        val conditions = FirebaseModelDownloadConditions.Builder()
                .requireWifi()
                .build()
        val remoteModel = FirebaseRemoteModel.Builder(HOSTED_MODEL_NAME)
                .enableModelUpdates(true)
                .setInitialDownloadConditions(conditions)
                .setUpdatesDownloadConditions(conditions)  // You could also specify
                // different conditions
                // for updates
                .build()
        val localModel = FirebaseLocalModel.Builder("asset")
                .setAssetFilePath(LOCAL_MODEL_ASSET).build()
        val manager = FirebaseModelManager.getInstance()
        manager.registerRemoteModel(remoteModel)
        manager.registerLocalModel(localModel)
        val modelOptions = FirebaseModelOptions.Builder()
                .setRemoteModelName(HOSTED_MODEL_NAME)
                .setLocalModelName("asset")
                .build()
        mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions)
    }

    private fun loadLabelList(context: Context): List<String> {
        val labelList = ArrayList<String>()
        BufferedReader(InputStreamReader(context.assets.open(LABEL_PATH))).use { reader ->
            reader.useLines {
                it.forEach { line -> labelList.add(line) }
            }
        }

        return labelList
    }

    suspend fun runTextRecognition(bitmap: Bitmap): FirebaseVisionText = suspendCoroutine { continuation ->
        val image = FirebaseVisionImage.fromBitmap(bitmap)
        val recognizer = FirebaseVision.getInstance()
                .onDeviceTextRecognizer
        recognizer.processImage(image)
                .addOnSuccessListener { texts -> continuation.resume(texts) }
                .addOnFailureListener { e -> continuation.resumeWithException(e) }
    }


    suspend fun runModelInference(bitmap: Bitmap): List<String> = suspendCoroutine { continuation ->
        val interpreter = mInterpreter
                ?: throw RuntimeException("Image classifier has not been initialized; Skipped.")

        val imgData = convertBitmapToByteBuffer(bitmap, bitmap.getWidth(), bitmap.getHeight())

        val inputs = FirebaseModelInputs.Builder().add(imgData).build()
        // Here's where the magic happens!!
        interpreter
                .run(inputs, mDataOptions)
                .addOnFailureListener { e ->
                    continuation.resumeWithException(e)
                }
                .continueWith(
                        Continuation<FirebaseModelOutputs, List<String>> { task ->
                            val labelProbArray = task.result!!
                                    .getOutput<Array<ByteArray>>(0)
                            val topLabels = getTopLabels(labelProbArray)
                            continuation.resume(topLabels)
                            topLabels
                        })
    }

    @Synchronized
    private fun convertBitmapToByteBuffer(
            bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val imgData = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE)
        imgData.order(ByteOrder.nativeOrder())
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
                true)
        imgData.rewind()
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0,
                scaledBitmap.width, scaledBitmap.height)
        // Convert the image to int points.
        var pixel = 0
        for (i in 0 until DIM_IMG_SIZE_X) {
            for (j in 0 until DIM_IMG_SIZE_Y) {
                val `val` = intValues[pixel++]
                imgData.put((`val` shr 16 and 0xFF).toByte())
                imgData.put((`val` shr 8 and 0xFF).toByte())
                imgData.put((`val` and 0xFF).toByte())
            }
        }
        return imgData
    }

    @Synchronized
    private fun getTopLabels(labelProbArray: Array<ByteArray>): List<String> {
        for (i in mLabelList.indices) {
            sortedLabels.add(
                    AbstractMap.SimpleEntry<String, Float>(mLabelList[i], (labelProbArray[0][i].toInt() and 0xff) / 255.0f))
            if (sortedLabels.size > RESULTS_TO_SHOW) {
                sortedLabels.poll()
            }
        }
        val result = ArrayList<String>()
        val size = sortedLabels.size
        for (i in 0 until size) {
            val label = sortedLabels.poll()
            result.add(label.key + ":" + label.value)
        }
        return result
    }
}
