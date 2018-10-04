using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
#if UNITY_UWP
using System;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.AI.MachineLearning;
using System.Linq;
using Windows.Graphics.Imaging;
using Windows.Storage.Streams;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Media;
#endif

public class WindowsML_Demo : MonoBehaviour {
    public Text text;

    private WebCamTexture webcam;
    private Texture2D Tex2d;
    private byte[] bytes = null;

    void Start()
    {
        // 学習モデルとラベルデータのローカルフォルダへの移動
        File.Copy(Application.streamingAssetsPath + "\\" + "model.onnx", Application.persistentDataPath + "\\" + "model.onnx", true);
        File.Copy(Application.streamingAssetsPath + "\\" + "Labels.json", Application.persistentDataPath + "\\" + "Labels.json", true);

        // Webカメラの取得と録画の開始
        webcam = new WebCamTexture("MN34150", 896, 504,15);
        Tex2d = new Texture2D(896, 504, TextureFormat.RGBA32, false);
        webcam.Play();

#if UNITY_UWP
        // WindowsMLのモデル読み込みの開始
        Task.Run(async () =>
        {
            await LoadModelAsync();
            await ObjectDetectation();
        });
#endif
        StartCoroutine(GetWebCamData());
    }

    private IEnumerator GetWebCamData()
    {
        while (true)
        {
            // Webカメラからの画像取得をbyte配列への変換
            Tex2d.SetPixels32(webcam.GetPixels32());
            yield return null;
            bytes = Tex2d.GetRawTextureData();
            yield return null;
        }
    }

#if UNITY_UWP
    private List<string> _labels = new List<string>();
    private LearningModel _model = null;
    private List<float> _outputVariableList = new List<float>();
    private LearningModelSession _session = null;
   
    private async Task LoadModelAsync()
    {
        try
        {
            // ラベルデータの読み込み
            var file = await Windows.Storage.ApplicationData.Current.LocalFolder.GetFileAsync("Labels.json");
            using (var inputStream = await file.OpenReadAsync())
            using (var classicStream = inputStream.AsStreamForRead())
            using (var streamReader = new StreamReader(classicStream))
            {
                string line = "";
                char[] charToTrim = { '\"', ' ' };
                while (streamReader.Peek() >= 0)
                {
                    line = streamReader.ReadLine();
                    line.Trim(charToTrim);
                    var indexAndLabel = line.Split(':');
                    if (indexAndLabel.Count() == 2)
                    {
                        _labels.Add(indexAndLabel[1]);
                    }
                }
            }

            // 学習モデルの読み込み
            var modelFile = await Windows.Storage.ApplicationData.Current.LocalFolder.GetFileAsync("model.onnx");
            _model = await LearningModel.LoadFromStorageFileAsync(modelFile);

            _session = new LearningModelSession(_model, new LearningModelDevice(LearningModelDeviceKind.Default));
        }
        catch (Exception ex)
        {
            _model = null;
            UnityEngine.WSA.Application.InvokeOnAppThread(() =>
            {
                text.text = ex.ToString();
            }, true);
        }
    }

    private async Task ObjectDetectation()
    {
        while (true)
        {
            if (bytes!=null)
            {
                // UnityのWebカメラは上下反転しているので入れ替え処理
                var buf = new byte[bytes.Length];
                for (int i = 0; i < 504; i++)
                {
                    for (int j = 0; j < 896; j++)
                    {
                        buf[(896 * (504 - 1 - i) + j) * 4 + 0] = bytes[(896 * i + j )* 4 + 0];
                        buf[(896 * (504 - 1 - i) + j) * 4 + 1] = bytes[(896 * i + j )* 4 + 1];
                        buf[(896 * (504 - 1 - i) + j) * 4 + 2] = bytes[(896 * i + j )* 4 + 2];
                        buf[(896 * (504 - 1 - i) + j) * 4 + 3] = bytes[(896 * i + j )* 4 + 3];
                    }
                }

                // 入力画像から物体認識を行う
                try
                {
                    SoftwareBitmap softwareBitmap;
                    softwareBitmap = new SoftwareBitmap(BitmapPixelFormat.Rgba8, 896, 504);
                    softwareBitmap.CopyFromBuffer(buf.AsBuffer());
                    bytes = null;
                    buf = null;
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                    VideoFrame inputFrame = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                    // WindowsMLに入力，出力形式を設定する
                    LearningModelBinding binding = new LearningModelBinding(_session);
                    ImageFeatureValue imageTensor = ImageFeatureValue.CreateFromVideoFrame(inputFrame);
                    binding.Bind("data_0", imageTensor);

                    // Process the frame with the model
                    var results = await _session.EvaluateAsync(binding, "");
                    var resultTensor = results.Outputs["softmaxout_1"] as TensorFloat;
                    var resultVector = resultTensor.GetAsVectorView();

                    // 認識結果から適合率の高い上位3位までを選択
                    List<float> topProbabilities = new List<float>() { 0.0f, 0.0f, 0.0f };
                    List<int> topProbabilityLabelIndexes = new List<int>() { 0, 0, 0 };
                    for (int i = 0; i < resultVector.Count(); i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            if (resultVector[i] > topProbabilities[j])
                            {
                                topProbabilityLabelIndexes[j] = i;
                                topProbabilities[j] = resultVector[i];
                                break;
                            }
                        }
                    }

                    // 結果を出力する
                    string message = "Predominant objects detected are:";
                    for (int i = 0; i < 3; i++)
                    {
                        message += $"\n{ _labels[topProbabilityLabelIndexes[i]]} with confidence of { topProbabilities[i]}";
                    }
                    softwareBitmap.Dispose();
                    UnityEngine.WSA.Application.InvokeOnAppThread(() =>
                    {
                        text.text = message;
                    }, true);
                }
                catch (Exception ex)
                {
                    UnityEngine.WSA.Application.InvokeOnAppThread(() =>
                    {
                        text.text = ex.ToString();
                    }, true);
                }
            }
        }
    }
#endif
}
