void main() {
  final Map<int, int> a = {};
  a[0] ??= calc(0);
  a[1] ??= calc(1);
  a[0] ??= calc(0);
  a[1] ??= calc(1);
}

int calc(int x) {
  print('calc($x)');
  return x * 2;
}
