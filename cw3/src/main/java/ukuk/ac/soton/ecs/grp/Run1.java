package ukuk.ac.soton.ecs.grp;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * OpenIMAJ Hello world!
 */
public class Run1 {
    public static void main(String[] args) throws FileSystemException {
        //GroupSampler.sample(App.trainingData, 5, false);
        DisplayUtilities.display("original", App.randomInstanceTest);
        DisplayUtilities.display("crop", cropImage(App.randomInstanceTest, 16));
        System.out.println(Arrays.toString(vectoriser(cropImage(App.randomInstanceTest, 16))));
        System.out.println(mapVector(App.trainingData));
        System.out.println(KNNClassifier(cropImage(App.randomInstanceTrain,16), mapVector(App.trainingData), 3));
    }

    //cropping image to a square about the centre
    private static FImage cropImage(FImage fullSized, int imageSize) {
        FImage squareImage;
        if (fullSized.getHeight() > fullSized.getWidth()) {
            squareImage = fullSized.extractCenter(fullSized.width, fullSized.width);
        } else {
            squareImage = fullSized.extractCenter(fullSized.height, fullSized.height);
        }
        return ResizeProcessor.resample(squareImage, imageSize, imageSize);
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
        String mostFrequent = null;
        int highestFrequency = 0;
        for (String name : frequencyMap.keySet()){
            if (frequencyMap.get(name) < highestFrequency){
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
