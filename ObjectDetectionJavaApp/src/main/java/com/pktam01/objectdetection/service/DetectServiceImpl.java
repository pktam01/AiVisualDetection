package com.pktam01.objectdetection.service;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.springframework.stereotype.Service;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.types.UInt8;

import com.google.protobuf.TextFormat;

import object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
import object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;

@Service
public class DetectServiceImpl implements DetectService {

	private final String labels_location = "labels//mscoco_label_map.pbtxt";
	private final String models_location = "models//ssd_inception_v2_coco_2017_11_17//saved_model";

	@Override
	public ArrayList<String> returnObjectType(String filename) throws Exception {
		
		ArrayList<String> stringlist= new ArrayList<String>();
		
		

		final String[] labels = loadLabels(labels_location);
		try (SavedModelBundle model = SavedModelBundle.load(models_location, "serve")) {
			List<Tensor<?>> outputs = null;
			System.out.println(filename);
			try (Tensor<UInt8> input = makeImageTensor(filename)) {
				outputs = model.session().runner().feed("image_tensor", input).fetch("detection_scores")
						.fetch("detection_classes").fetch("detection_boxes").run();
			}
			try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
					Tensor<Float> classesT = outputs.get(1).expect(Float.class);
					Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
				// All these tensors have:
				// - 1 as the first dimension
				// - maxObjects as the second dimension
				// While boxesT will have 4 as the third dimension (2 sets of (x, y)
				// coordinates).
				// This can be verified by looking at scoresT.shape() etc.
				int maxObjects = (int) scoresT.shape()[1];
				float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
				float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
				float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
				// Print all objects whose score is at least 0.5.
				System.out.printf("* %s\n", filename);
				boolean foundSomething = false;
				for (int i = 0; i < scores.length; ++i) {
					if (scores[i] < 0.5) {
						continue;
					}
					foundSomething = true;
					stringlist.add(String.format("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]));
					System.out.printf("\tFound %-20s (score: %.4f)\n", labels[(int) classes[i]], scores[i]);
				}
				if (!foundSomething) {
					System.out.println("No objects detected with a high enough score.");
				}
			}

			return stringlist;
		}
	}

	private String[] loadLabels(String filename) throws Exception {
		String text = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8);
		StringIntLabelMap.Builder builder = StringIntLabelMap.newBuilder();
		TextFormat.merge(text, builder);
		StringIntLabelMap proto = builder.build();
		int maxId = 0;
		for (StringIntLabelMapItem item : proto.getItemList()) {
			if (item.getId() > maxId) {
				maxId = item.getId();
			}
		}
		String[] ret = new String[maxId + 1];
		for (StringIntLabelMapItem item : proto.getItemList()) {
			ret[item.getId()] = item.getDisplayName();
		}
		return ret;
	}

	private static void bgr2rgb(byte[] data) {
		for (int i = 0; i < data.length; i += 3) {
			byte tmp = data[i];
			data[i] = data[i + 2];
			data[i + 2] = tmp;
		}
	}

	private static Tensor<UInt8> makeImageTensor(String filename) throws IOException {
		BufferedImage img = ImageIO.read(new File(filename));
		if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
			throw new IOException(String.format(
					"Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
					img.getType(), filename));
		}
		byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
		// ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
		bgr2rgb(data);
		final long BATCH_SIZE = 1;
		final long CHANNELS = 3;
		long[] shape = new long[] { BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS };
		return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
	}
}
