package ukuk.ac.soton.ecs.grp;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

public class App {
    static String trainingPath = "C:\\Users\\jplam\\Desktop\\Comp Vision\\CW3\\scene-recognition\\cw3\\src\\main\\java\\ukuk\\ac\\soton\\ecs\\grp\\training.zip";
    static String testingPath = "C:\\Users\\jplam\\Desktop\\Comp Vision\\CW3\\scene-recognition\\cw3\\src\\main\\java\\ukuk\\ac\\soton\\ecs\\grp\\training.zip";

    static VFSListDataset<FImage> trainingData;
    static VFSListDataset<FImage> testingData;

    static {
        try {
            trainingData = new VFSListDataset<FImage>("zip:" + trainingPath, ImageUtilities.FIMAGE_READER);
            testingData = new VFSListDataset<FImage>("zip:" + testingPath, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
    }

    static FImage randomInstanceTrain = trainingData.getRandomInstance();
    static FImage randomInstanceTest = testingData.getRandomInstance();
}
