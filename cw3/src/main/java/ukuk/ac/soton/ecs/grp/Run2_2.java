package ukuk.ac.soton.ecs.grp;


import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;

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

        return vectors.toArray();

        /*float[][] arrayVectors= new float[vectors.size()][vectors.get(0).length];

        for (int i = 0; i < vectors.size(); i++){
            arrayVectors[i] = vectors.get(i);
        }
        return arrayVectors;*/
    }

    private static void KMeansClustering(float[][] floatPatches){
        final FloatKMeans kMeans =  FloatKMeans.createExact(500);
        FloatCentroidsResult result = kMeans.cluster(floatPatches);
        /*float[][] centroids = result.centroids;

        for (float[] fs : centroids) {
            System.out.println(Arrays.toString(fs));
        }*/
    }
}
