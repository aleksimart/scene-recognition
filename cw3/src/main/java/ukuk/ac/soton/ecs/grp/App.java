package ukuk.ac.soton.ecs.grp;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

/**
 * OpenIMAJ Hello world!
 *
 */
public class App {
    public static void main( String[] args ) throws FileSystemException {
        String trainingPath = "C:\\Users\\jplam\\Desktop\\Comp Vision\\CW3\\scene-recognition\\cw3\\src\\main\\java\\ukuk\\ac\\soton\\ecs\\grp\\training.zip";
        String testingPath = "C:\\Users\\jplam\\Desktop\\Comp Vision\\CW3\\scene-recognition\\cw3\\src\\main\\java\\ukuk\\ac\\soton\\ecs\\grp\\training.zip";

        VFSListDataset<FImage> trainingData =
                new VFSListDataset<FImage>("zip:" + trainingPath, ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData =
                new VFSListDataset<FImage>("zip:" + testingPath, ImageUtilities.FIMAGE_READER);

        DisplayUtilities.display("random training",trainingData.getRandomInstance());
        DisplayUtilities.display("random testing", testingData.getRandomInstance());

        //change to demonstrate PRs
    }
}
