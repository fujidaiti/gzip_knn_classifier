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

    test('computeNCD', () {
      // Samples from AG_NEWS dataset
      const testText =
          r'''Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters - Soaring crude prices plus worries\about the economy and the outlook for earnings are expected to\hang over the stock market next week during the depth of the\summer doldrums.''';
      const trainTexts = [
        r'''Stocks End Up, But Near Year Lows (Reuters) Reuters - Stocks ended slightly higher on Friday\but stayed near lows for the year as oil prices surged past  #36;46\a barrel, offsetting a positive outlook from computer maker\Dell Inc. (DELL.O)''',
        r'''Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again.''',
        r'''Iraq Halts Oil Exports from Main Southern Pipeline (Reuters) Reuters - Authorities have halted oil export\flows from the main pipeline in southern Iraq after\intelligence showed a rebel militia could strike\infrastructure, an oil official said on Saturday.''',
        r'''Oil and Economy Cloud Stocks' Outlook  NEW YORK (Reuters) - Soaring crude prices plus worries  about the economy and the outlook for earnings are expected to  hang over the stock market next week during the depth of the  summer doldrums.''',
        r'''Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group,\which has a reputation for making well-timed and occasionally\controversial plays in the defense industry, has quietly placed\its bets on another part of the market.''',
        r'''Money Funds Fell in Latest Week (AP) AP - Assets of the nation's retail money market mutual funds fell by  #36;1.17 billion in the latest week to  #36;849.98 trillion, the Investment Company Institute said Thursday.''',
        r'''Fed minutes show dissent over inflation (USATODAY.com) USATODAY.com - Retail sales bounced back a bit in July, and new claims for jobless benefits fell last week, the government said Thursday, indicating the economy is improving from a midsummer slump.''',
        r'''Wall St. Bears Claw Back Into the Black  NEW YORK (Reuters) - Short-sellers, Wall Street's dwindling  band of ultra-cynics, are seeing green again.''',
        r'''Safety Net (Forbes.com) Forbes.com - After earning a PH.D. in Sociology, Danny Bazil Riley started to work as the general manager at a commercial real estate firm at an annual base salary of  #36;70,000. Soon after, a financial planner stopped by his desk to drop off brochures about insurance benefits available through his employer. But, at 32, "buying insurance was the furthest thing from my mind," says Riley.''',
        r'''Oil prices soar to all-time record, posing new menace to US economy (AFP) AFP - Tearaway world oil prices, toppling records and straining wallets, present a new economic menace barely three months before the US presidential elections.''',
      ];
      // Calculated NCDs between `testText` and each of `trainTexts` using the author's original code.
      // Link: https://github.com/bazingagin/npc_gzip/blob/b05a7bb80f07b7c32edf80e34bfc6eedf637eacd/original_codebase/experiments.py#L24-L26
      const expectedNCDs = [
        0.6945812807881774,
        0.6949152542372882,
        0.7076923076923077,
        0.19672131147540983,
        0.6908212560386473,
        0.7471910112359551,
        0.725,
        0.711864406779661,
        0.8440677966101695,
        0.702247191011236
      ];

      final classifier = KnnGzipClassifier(
        trainDataset: [(label: '', text: '')],
        reducer: const MajorityVoting(),
        compressor: Gzip(),
      );

      assert(trainTexts.length == expectedNCDs.length);
      for (var i = 0; i < trainTexts.length; ++i) {
        final ncd = classifier.computeNCD(trainTexts[i], testText);
        const tolerance = 0.02;
        expect(ncd, closeTo(expectedNCDs[i], tolerance));
      }
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
  int compressedLength(String text) => text.length;
}
