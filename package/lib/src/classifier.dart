import 'dart:math';

import 'package:knn_gzip_classifier/src/compressors.dart';
import 'package:meta/meta.dart';

typedef TrainDataset = List<({String text, String label})>;
typedef PredictionResult = List<({String label, double metric})>;

class KnnGzipClassifier<T> {
  KnnGzipClassifier({
    required this.trainDataset,
    required this.reducer,
    this.k = 2,
    Compressor? compressor,
  })  : compressor = compressor ?? Gzip(),
        assert(trainDataset.isNotEmpty),
        assert(k > 0);

  final int k;
  final TrainDataset trainDataset;
  final Compressor compressor;
  final Reducer<T> reducer;

  T predict(String test) {
    assert(test.isNotEmpty);
    return reducer.reduce(findNearestNeighbors(test));
  }

  @visibleForTesting
  PredictionResult findNearestNeighbors(String test) {
    final sortedTrainData = trainDataset.map((train) {
      final ncd = computeNCD(test, train.text);
      return (label: train.label, metric: ncd);
    }).toList(growable: false);
    sortedTrainData.sort((t1, t2) => t1.metric.compareTo(t2.metric));
    return sortedTrainData.take(k).toList(growable: false);
  }

  /// Computes the Normalized Comression Distance (NCD) between [x] and [y].
  @visibleForTesting
  double computeNCD(String x, String y) {
    assert(x.isNotEmpty && y.isNotEmpty);
    final xCmp = compressor.compress(x).length;
    final yCmp = compressor.compress(y).length;
    final xyCmp = compressor.compress('$x $y').length;
    return (xyCmp - min(xCmp, yCmp)) / max(xCmp, yCmp);
  }
}

abstract class Reducer<T> {
  T reduce(PredictionResult result);
}

class MajorityVoting implements Reducer<String> {
  const MajorityVoting();

  @override
  String reduce(PredictionResult result) {
    assert(result.isNotEmpty);
    final occurrence = <String, int>{};
    for (final (:label, metric: _) in result) {
      occurrence[label] = (occurrence[label] ?? 0) + 1;
    }
    return occurrence.keys
        .reduce((a, b) => occurrence[a]! > occurrence[b]! ? a : b);
  }
}
