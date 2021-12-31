package ukuk.ac.soton.ecs.grp;


import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Run2_2 {


    public static void main (String[] args){
        KMeansClustering(getPatches(App.randomInstanceTest, 4, 8));
    }

    private static float[][] getPatches(FImage inputImage, int frequency, int patchsize){
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
        return vectors.toArray(arr);
    }

    private static void KMeansClustering(float[][] floatPatches){
        final FloatKMeans kMeans =  FloatKMeans.createExact(500);
        FloatCentroidsResult result = kMeans.cluster(floatPatches);
        float[][] centroids = result.centroids;

        for (float[] fs : centroids) {
            System.out.println(Arrays.toString(fs));
        }
    }

    static class BOVWExtractor implements FeatureExtractor<DoubleFV, FImage> {

        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public BOVWExtractor(HardAssigner<byte[], float[], IntFloatPair> assigner) {
            this.assigner = assigner;
        }

        @Override
        public DoubleFV extractFeature(FImage object) {
            K
        }
    }
}
