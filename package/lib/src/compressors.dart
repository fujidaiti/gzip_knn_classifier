import 'dart:convert';
import 'dart:io' as io;

abstract class Compressor {
  String compress(String text);
}

class Gzip implements Compressor {
  final Map<String, String> _cache = {};

  @override
  String compress(String text) {
    _cache[text] ??= base64.encode(io.gzip.encode(text.codeUnits));
    return _cache[text]!;
  }
}
