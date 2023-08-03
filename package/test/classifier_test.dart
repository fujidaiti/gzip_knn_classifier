import 'package:knn_gzip_classifier/src/classifier.dart';
import 'package:knn_gzip_classifier/src/compressors.dart';
import 'package:test/test.dart';

void main() {
  group('KnnGzipClassifier', () {
    test('findNearestNeighbors', () {
      final classifier = _TestClassifier(
        k: 3,
        trainDataset: [
          (text: 'a2', label: 'A'),
          (text: 'a3_', label: 'A'),
          (text: 'b3_', label: 'B'),
          (text: 'c4__', label: 'C'),
        ],
      );

      final result = classifier.findNearestNeighbors('a');
      expect(result.length, 3);
      expect(result, [
        (label: 'A', metric: 1.0),
        (label: 'A', metric: 2.0),
        (label: 'B', metric: 2.0),
      ]);
    });
  });

  group('Reducers', () {
    test('MajorityVoting', () {
      final reducer = MajorityVoting();
      expect(
          reducer.reduce([
            // Doesn't care about "metric"s
            (label: 'A', metric: 0),
            (label: 'B', metric: 0),
            (label: 'A', metric: 0),
            (label: 'C', metric: 0),
          ]),
          'A');
    });
  });
}

class _TestClassifier extends KnnGzipClassifier {
  _TestClassifier({
    required super.trainDataset,
    required super.k,
  }) : super(
          reducer: const MajorityVoting(),
          compressor: const _TestCompressor(),
        );

  @override
  double computeNCD(String x, String y) =>
      (x.length - y.length).abs().toDouble();
}

class _TestCompressor implements Compressor {
  const _TestCompressor();

  @override
  String compress(String text) => text;
}
