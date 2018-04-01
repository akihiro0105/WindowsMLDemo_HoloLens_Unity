using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;
#if UNITY_UWP
using System;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.AI.MachineLearning.Preview;
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
    // Use this for initialization
    void Start()
    {
        // 学習モデルとラベルデータのローカルフォルダへの移動
        File.Copy(Application.streamingAssetsPath + "\\" + "SqueezeNet.onnx", Application.persistentDataPath + "\\" + "SqueezeNet.onnx", true);
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

    // Update is called once per frame
    void Update () {

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
    private LearningModelPreview _model = null;
    private ImageVariableDescriptorPreview _inputImageDescription;
    private TensorVariableDescriptorPreview _outputTensorDescription;
    private List<float> _outputVariableList = new List<float>();

    private async Task LoadModelAsync()
    {
        try
        {
            // ラベルデータの読み込み
            var file = await ApplicationData.Current.LocalFolder.GetFileAsync("Labels.json");
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
            var modelFile = await ApplicationData.Current.LocalFolder.GetFileAsync("SqueezeNet.onnx");
            _model = await LearningModelPreview.LoadModelFromStorageFileAsync(modelFile);

            List<ILearningModelVariableDescriptorPreview> inputFeatures = _model.Description.InputFeatures.ToList();
            List<ILearningModelVariableDescriptorPreview> outputFeatures = _model.Description.OutputFeatures.ToList();
            _inputImageDescription = inputFeatures.FirstOrDefault(feature => feature.ModelFeatureKind == LearningModelFeatureKindPreview.Image) as ImageVariableDescriptorPreview;
            _outputTensorDescription = outputFeatures.FirstOrDefault(feature => feature.ModelFeatureKind == LearningModelFeatureKindPreview.Tensor) as TensorVariableDescriptorPreview;
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
                    LearningModelBindingPreview binding = new LearningModelBindingPreview(_model as LearningModelPreview);
                    binding.Bind(_inputImageDescription.Name, inputFrame);
                    binding.Bind(_outputTensorDescription.Name, _outputVariableList);

                    // Process the frame with the model
                    LearningModelEvaluationResultPreview results = await _model.EvaluateAsync(binding, "test");
                    List<float> resultProbabilities = results.Outputs[_outputTensorDescription.Name] as List<float>;

                    // 認識結果から適合率の高い上位3位までを選択
                    List<float> topProbabilities = new List<float>() { 0.0f, 0.0f, 0.0f };
                    List<int> topProbabilityLabelIndexes = new List<int>() { 0, 0, 0 };
                    for (int i = 0; i < resultProbabilities.Count(); i++)
                    {
                        for (int j = 0; j < 3; j++)
                        {
                            if (resultProbabilities[i] > topProbabilities[j])
                            {
                                topProbabilityLabelIndexes[j] = i;
                                topProbabilities[j] = resultProbabilities[i];
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
