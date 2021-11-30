package ukuk.ac.soton.ecs.grp;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
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

/**
 * OpenIMAJ Hello world!
 *
 */
public class Run1 {
    public static void main( String[] args ) throws FileSystemException {
        //GroupSampler.sample(App.trainingData, 5, false);
        DisplayUtilities.display("original", App.randomInstanceTest);
        DisplayUtilities.display("crop", cropImage(App.randomInstanceTest,16));
        System.out.println(Arrays.toString(vectoriser(cropImage(App.randomInstanceTest, 16))));
    }

    //cropping image to a square about the centre
    private static FImage cropImage(FImage fullSized, int imageSize){
        FImage squareImage;
        if(fullSized.getHeight()>fullSized.getWidth()){
            squareImage = fullSized.extractCenter(fullSized.width, fullSized.width);
        } else {
            squareImage = fullSized.extractCenter(fullSized.height, fullSized.height);
        }
        return ResizeProcessor.resample(squareImage, imageSize, imageSize);
    }

    private static float[] vectoriser(FImage originalImage){
        float[][] pixels = originalImage.pixels;
        float[] concatRow = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++){
            float concatVal = 0;
            float[] row = pixels[i];
            for (float val : row){
                concatVal += val;
            }
            concatRow[i] = concatVal;
        }
        return concatRow;
    }
}
