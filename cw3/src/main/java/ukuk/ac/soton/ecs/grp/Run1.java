package ukuk.ac.soton.ecs.grp;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;

import javax.sound.midi.SysexMessage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * OpenIMAJ Hello world!
 */
public class Run1 {
    public static void main(String[] args) throws FileSystemException {
        GroupedRandomSplitter<String, FImage> splits =
                new GroupedRandomSplitter<String,FImage>(App.trainingData ,15 , 0, 15);
        //DisplayUtilities.display("original", App.randomInstanceTest);
        //DisplayUtilities.display("crop", cropImage(App.randomInstanceTest, 16));
        //System.out.println(Arrays.toString(vectoriser(cropImage(App.randomInstanceTest, 16))));
        Map<String, float[][]> trainingVectors = mapVector(App.trainingData);
        //Map<String, float[][]> trainingVectors = mapVector(splits.getTrainingDataset());
        //System.out.println(Arrays.toString(App.testingData.getFileObjects()));
        int incorrect = 0;
        int correct = 0;
        try {
            PrintWriter printWriter = new PrintWriter(new File("run1.txt"));
            for (int i = 0; i < App.testingData.size(); i++){
                //System.out.println(image.getHeight());
                //System.out.println(image.getWidth());
                printWriter.println(App.testingData.getID(i).substring(8) +  " " + KNNClassifier(cropImage(App.testingData.get(i), 16), trainingVectors, 5));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        /*for (Map.Entry<String, ListDataset<FImage>> testImage: splits.getTestDataset().entrySet() ){
            for (FImage randomInstance : testImage.getValue()) {
                System.out.println(testImage.getKey());
                        String prediction = KNNClassifier(cropImage(randomInstance, 16), trainingVectors, 5);
                System.out.println(prediction + " " + testImage.getKey());
                if (prediction.equals(testImage.getKey()))
                    correct++;
                else
                    incorrect++;
            }
        }
        System.out.println(correct + " " + incorrect);*/


    }
    //cropping image to a square about the centre
    private static FImage cropImage(FImage fullSized, int imageSize) {
        FImage squareImage;
        if (fullSized.getHeight() > fullSized.getWidth()) {
            squareImage = fullSized.extractCenter(fullSized.width, fullSized.width);
        } else {
            squareImage = fullSized.extractCenter(fullSized.height, fullSized.height);
        }
        FImage croppedImage = ResizeProcessor.resample(squareImage, imageSize, imageSize);
        return croppedImage.subtract(averageFloat(croppedImage)).normalise();
    }

    public static float averageFloat(FImage inputImage){
        float[][] pixels = inputImage.pixels;
        float sum = 0;
        for (float[] pixel : pixels) {
            for (int y = 0; y < pixels[0].length; y++) {
                sum += pixel[y];
            }
        }
        return sum / (pixels.length * pixels[1].length);
    }

    private static Map<String, float[][]> mapVector(GroupedDataset<String, ListDataset<FImage>, FImage> groupedData) {
        Map<String, float[][]> output = new HashMap<String, float[][]>();
        for (String groupName : groupedData.getGroups()) {
            int trainingSize = groupedData.get(groupName).size();
            float[][] featureList = new float[trainingSize][16];
            for (int i = 0; i < trainingSize; i++) {

                float[] vectoriser = vectoriser(cropImage(groupedData.get(groupName).get(i),16));
                featureList[i] = vectoriser;
            }
            output.put(groupName, featureList);
        }
        return output;
    }

    private static Map<String, float[][]> mapVector(VFSGroupDataset<FImage> groupedData) {
        Map<String, float[][]> output = new HashMap<String, float[][]>();
        for (String groupName : groupedData.getGroups()) {
            int trainingSize = groupedData.get(groupName).size();
            float[][] featureList = new float[trainingSize][16];
            for (int i = 0; i < trainingSize; i++) {

                float[] vectoriser = vectoriser(cropImage(groupedData.get(groupName).get(i),16));
                featureList[i] = vectoriser;
            }
            output.put(groupName, featureList);
        }
        return output;
    }

    private static String KNNClassifier(FImage testImage, Map<String, float[][]> trainingMap, int k) {
        HashMap<Float, String> distanceMap = new HashMap<Float, String>();
        float highestMinDistance = Float.MIN_VALUE;
        float[] testVector = vectoriser(testImage);
        //System.out.println(Arrays.toString(testVector));
        for (String groupName : trainingMap.keySet()) {
            for (float[] trainingVector : trainingMap.get(groupName)) {
                float distance = distance(trainingVector, testVector);
                if (distanceMap.size() < k) {
                    distanceMap.put(distance, groupName);
                    if (distance > highestMinDistance)
                        highestMinDistance = distance;
                } else {
                    if (distance < highestMinDistance) {
                        distanceMap.remove(highestMinDistance);
                        distanceMap.put(distance, groupName);
                        highestMinDistance = maxArr(distanceMap);
                    }
                }
            }
        }
        //System.out.println(distanceMap.toString());

        float minVal = Float.MAX_VALUE;
        for (Float value : distanceMap.keySet()){
            if (minVal > value){
                minVal = value;
            }
        }
        HashMap<String, Integer> frequencyMap = new HashMap<String,Integer>();
        for (String name : distanceMap.values()) {
            if (frequencyMap.containsKey(name)) {
                frequencyMap.replace(name, frequencyMap.get(name) + 1);
            } else {
                frequencyMap.put(name, 1);
            }
        }
        //System.out.println (frequencyMap.toString());
        String mostFrequent = null;
        int highestFrequency = 0;
        for (String name : frequencyMap.keySet()){
            if (frequencyMap.get(name) > highestFrequency){
                mostFrequent = name;
                highestFrequency = frequencyMap.get(name);
            }
        }
        if (highestFrequency <= 1){
            return distanceMap.get(minVal);
        } else {
            return mostFrequent;
        }
    }

    private static float maxArr(Map<Float, String> inputList) {
        float maxVal = Float.MIN_VALUE;
        for (float val : inputList.keySet()) {
            if (maxVal < val) {
                maxVal = val;
            }
        }
        return maxVal;
    }

    private static float distance(float[] v1, float[] v2) {
        float sum = 0;
        for (int i = 0; i < v1.length; i++) {
            sum += Math.pow((v1[i] - v2[i]), 2);
        }
        return (float) Math.pow(sum, 0.5);
    }

    private static float[] vectoriser(FImage originalImage) {
        float[][] pixels = originalImage.pixels;
        float[] concatRow = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            float concatVal = 0;
            float[] row = pixels[i];
            for (float val : row) {
                concatVal += val;
            }
            concatRow[i] = concatVal;
        }
        return concatRow;
    }
}
