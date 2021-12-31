package ukuk.ac.soton.ecs.grp;
import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.BagOfWordsFeatureExtractor;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;
import java.util.ArrayList;

import java.util.List;

public class Run2 {
    public static void main(String[] args){
        libLinAnn();
    }

    private static float[][] sampler(FImage image, int patchSize, int frequency){
        float[][] pixels = image.pixels;
        ArrayList<float[]> patchList =  new ArrayList<float[]>();
        for (int x = 0; x < pixels.length - patchSize; x += frequency ){
            for (int y = 0; y < pixels[0].length - patchSize; y += frequency){
                float[][] patch =  new float[patchSize][patchSize];
                extractPatch(patchSize, pixels, x, y, patch);
                float[] flatPatch = flatten(patch,patchSize);
                patchList.add(flatPatch);
            }
        }
        float[][] output = new float[patchList.size()][patchSize*patchSize];
        for (int i = 0;i<patchList.size();i++) {
            output[i] = patchList.get(i);
        }
        return output;
    }

    private static float[] flatten(float[][] patch, int patchSize) {
        float[] output = new float[patchSize * patchSize];
        int itr = 0;
        for (float[] row:patch) {
            for (float val : row) {
                output[itr] = val;
                itr++;
            }
        }
        return output;
    }


    private static void extractPatch(int patchSize, float[][] pixels, int x, int y, float[][] patch) {
        for (int i = 0; i < patchSize; i++){
            for (int z = 0; z < patchSize; z++){
                patch[z][i] = pixels[x+z][i+ y];
            }
        }
    }

    private static void libLinAnn(){
        FloatKMeans clusters = FloatKMeans.createExact(500);
        FloatCentroidsResult result = clusters.cluster(sampler(App.randomInstanceTest, 8, 4));
        final HardAssigner<float[], ?, ?> assigner = result.defaultHardAssigner();
        BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>(assigner);
        /*LiblinearAnnotator<FImage, String >  ann = new LiblinearAnnotator<FImage, String> (bovw, LiblinearAnnotator.Mode.MULTILABEL,
                SolverType.L2R_L2LOSS_SVC, 1.0, 0.0001);*/
    }

}
