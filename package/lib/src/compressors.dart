import 'dart:io' as io;

abstract class Compressor {
  int compressedLength(String text);
}

class Gzip implements Compressor {
  const Gzip();

  @override
  int compressedLength(String text) {
    // _cache[text] ??= io.gzip.encode(text.codeUnits);
    // return _cache[text]!;
    return io.gzip.encode(text.codeUnits).length;
  }
}
