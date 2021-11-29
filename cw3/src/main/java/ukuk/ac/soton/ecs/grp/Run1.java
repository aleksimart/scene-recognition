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
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.image.typography.hershey.HersheyFont;

/**
 * OpenIMAJ Hello world!
 *
 */
public class Run1 {
    public static void main( String[] args ) throws FileSystemException {

        DisplayUtilities.display("crop", cropImage(App.randomInstanceTest,16));

    }
    //cropping image to a square about the centre
    private static FImage cropImage(FImage fullSized, int imageSize){
        return fullSized.extractCenter(imageSize,imageSize);
    }
}
