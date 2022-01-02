package ukuk.ac.soton.ecs.grp;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101;

public class App {
    static String trainingPath = "C:\\Users\\jplam\\Desktop\\Comp Vision\\CW3\\scene-recognition\\cw3\\src\\main\\java\\ukuk\\ac\\soton\\ecs\\grp\\training.zip";
    static String testingPath = "C:\\Users\\jplam\\Desktop\\Comp Vision\\CW3\\scene-recognition\\cw3\\src\\main\\java\\ukuk\\ac\\soton\\ecs\\grp\\testing.zip";

    static VFSGroupDataset<FImage> trainingData;
    static VFSListDataset<FImage> testingData;
    static GroupedRandomSplitter<String, FImage> splits;

    static {
        try {
            //GroupedDataset<String, VFSListDataset<Record<FImage>>, Record<FImage>>
            trainingData = new VFSGroupDataset<FImage>("zip:" + trainingPath + "!training/", ImageUtilities.FIMAGE_READER);
            splits = new GroupedRandomSplitter<String, FImage>(trainingData, 15, 0, 15);

            testingData = new VFSListDataset<FImage>("zip:" + testingPath, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
    }

    static FImage randomInstanceTrain = trainingData.getRandomInstance();
    static FImage randomInstanceTest = testingData.getRandomInstance();

}
