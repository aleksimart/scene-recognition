package ukuk.ac.soton.ecs.grp;


import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Run2_2 {


    public static void main (String[] args){
        List<float[]> patches = getPatches(App.randomInstanceTest, 4, 8);
        HardAssigner<float[], float[], IntFloatPair> assigner = KMeansClustering(patches.toArray(new float[patches.size()][]));
        BOVWExtractor extractor = new BOVWExtractor(assigner);
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<FImage, String>(
               extractor , LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001 );
        annotator.train( App.trainingData );
    }

    private static List<float[]> getPatches(FImage inputImage, int frequency, int patchsize){
        RectangleSampler sampler = new RectangleSampler(inputImage, frequency, frequency, patchsize, patchsize);
        List<Rectangle> rectangles = sampler.allRectangles();

        ArrayList<float[]> vectors = new ArrayList<>();

        for (Rectangle rectangle: rectangles){
            FImage imagePatch = inputImage.extractROI(rectangle);
            imagePatch = imagePatch.subtract(Run1.averageFloat(imagePatch));
            imagePatch = imagePatch.normalise();
            vectors.add(imagePatch.getFloatPixelVector());
        }
        float[][] arr = new float[vectors.size()][];
        return vectors;
    }

    private static HardAssigner<float[], float[], IntFloatPair> KMeansClustering(float[][] floatPatches){
        final FloatKMeans kMeans =  FloatKMeans.createExact(500);
        FloatCentroidsResult result = kMeans.cluster(floatPatches);
        HardAssigner<float[], float[], IntFloatPair> hardAssigner =  result.defaultHardAssigner();
        return hardAssigner;
    }

    static class BOVWExtractor implements FeatureExtractor<SparseIntFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;

        BOVWExtractor(HardAssigner<float[], float[], IntFloatPair> assigner){
            this.assigner = assigner;
        }

        @Override
        public SparseIntFV extractFeature(FImage object) {
            List<float[]> imageFeatureVectors = getPatches(object, 4,8 );
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<float[]>( assigner );
            return bovw.aggregateVectorsRaw( imageFeatureVectors );
        }

    }
}
