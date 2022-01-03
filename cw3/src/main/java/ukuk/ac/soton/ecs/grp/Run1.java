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
        //testing purposes
        //GroupedRandomSplitter<String, FImage> splits =
        //        new GroupedRandomSplitter<String,FImage>(App.trainingData ,15 , 0, 15);
        //DisplayUtilities.display("original", App.randomInstanceTest);
        //DisplayUtilities.display("crop", cropImage(App.randomInstanceTest, 16));
        //System.out.println(Arrays.toString(vectoriser(cropImage(App.randomInstanceTest, 16))));

        //Mapping the vector function across all of the training data.
        Map<String, float[][]> trainingVectors = mapVector(App.trainingData);

        //Map<String, float[][]> trainingVectors = mapVector(splits.getTrainingDataset());
        //System.out.println(Arrays.toString(App.testingData.getFileObjects()));
        //int incorrect = 0;
        //int correct = 0;

        /*
        Prints to run1.txt, with the classifcation results of KNN using parameters set there.
         */
        try {
            PrintWriter printWriter = new PrintWriter(new File("run1.txt"));
            for (int i = 0; i < App.testingData.size(); i++){
                printWriter.println(App.testingData.getID(i).substring(8) +  " " + KNNClassifier(cropImage(App.testingData.get(i), 16), trainingVectors, 5));
            }
            printWriter.flush();
            printWriter.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        /*
        code used for ascertaining our accuracy from the training data
         */
        /*
        for (Map.Entry<String, ListDataset<FImage>> testImage: splits.getTestDataset().entrySet() ){
            for (FImage randomInstance : testImage.getValue()) {
                System.out.println(testImage.getKey());
                        String prediction = KNNClassifier(cropImage(randomInstance, 16), trainingVectors, 39);
                System.out.println(prediction + " " + testImage.getKey());
                if (prediction.equals(testImage.getKey()))
                    correct++;
                else
                    incorrect++;
            }
        }
        System.out.println(correct + " " + incorrect);
        */
    }

    /**
     * Crops the image to a square then reduces the resolution to the specified image size, normalized and 0 mean.
     * @param fullSized: FImage, the full sized input image
     * @param imageSize: int, the x by x size of the cropped image that you want to output
     * @return FImage of the cropped image with the specfied dimensions
     */
    private static FImage cropImage(FImage fullSized, int imageSize) {
        FImage squareImage;
        if (fullSized.getHeight() > fullSized.getWidth()) {
            squareImage = fullSized.extractCenter(fullSized.width, fullSized.width);
        } else {
            squareImage = fullSized.extractCenter(fullSized.height, fullSized.height);
        }
        //resampling to smaller image and normalizing/ 0 meaning the image
        FImage croppedImage = ResizeProcessor.resample(squareImage, imageSize, imageSize);
        return croppedImage.subtract(averageFloat(croppedImage)).normalise();
    }

    /**
     * calculates the average value of a float in an FImage
     * @param inputImage FImage, to find the average float of the pixel values.
     * @return float, the average value of pixels in the input image
     */
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

    //overloaded version of mapvector for accuracy assessing
    /*
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
    }*/

    /**
     * Maps the vectoriser across a VFS GroupDataset
     * @param groupedData: This is the VFSGroupDataset that is used for training and be turned into vectors
     * @return A map of the vectors with their string key (these are the classes of images in the set)
     */
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

    /**
     *
     * @param testImage: FImage, This is the image to classify into the categories
     * @param trainingMap: Map<String,float[][]>, This is all of the float vectors from the training set with their category labels
     * @param k: int, the value of k used for the k nearest algorithm
     * @return String, the classifier result using KNN and taking the closest when it is a draw
     */
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
        //calculating which is the most frequent in the k nearest
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

    /**
     * Goes through the list and checks what is the maximum key value
     * @param inputList Map<Float,String>, The input of which you wish to calculate the maximum key value from
     * @return the float of the largest key value
     */
    private static float maxArr(Map<Float, String> inputList) {
        float maxVal = Float.MIN_VALUE;
        for (float val : inputList.keySet()) {
            if (maxVal < val) {
                maxVal = val;
            }
        }
        return maxVal;
    }

    /**
     * Uses Pythagoras to calculate the distance in 2D space
     * @param v1 float value of the pixel from one image
     * @param v2 float value fo the pixel from another image
     * @return the distance between these two vectors
     */
    private static float distance(float[] v1, float[] v2) {
        float sum = 0;
        for (int i = 0; i < v1.length; i++) {
            sum += Math.pow((v1[i] - v2[i]), 2);
        }
        return (float) Math.pow(sum, 0.5);
    }

    /**
     * Method turns an image from an FImage into a representing vector that is packed.
     * @param originalImage FImage, image to be vectorised
     * @return float[], The pixel values packed into a vector by concatenating each image row
     */
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
