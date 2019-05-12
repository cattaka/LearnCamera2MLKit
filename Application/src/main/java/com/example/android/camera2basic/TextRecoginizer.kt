package com.example.android.camera2basic

import android.content.Context
import android.graphics.Bitmap
import com.google.firebase.ml.common.modeldownload.FirebaseLocalModel
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel
import com.google.firebase.ml.custom.FirebaseModelDataType
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions
import com.google.firebase.ml.custom.FirebaseModelInterpreter
import com.google.firebase.ml.custom.FirebaseModelOptions
import com.google.firebase.ml.vision.FirebaseVision
import com.google.firebase.ml.vision.common.FirebaseVisionImage
import com.google.firebase.ml.vision.text.FirebaseVisionText
import java.io.BufferedReader
import java.io.InputStreamReader
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

        private val HOSTED_MODEL_NAME = "cloud_model_1"
        private val LOCAL_MODEL_ASSET = "mobilenet_v1_1.0_224_quant.tflite"
        private val LABEL_PATH = "labels.txt"

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
}
