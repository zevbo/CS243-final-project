
size_t microtime() {
  // struct timeb t;
  // ftime(&t);
  struct timespec t = {0, 0};
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000000 + t.tv_nsec / 1000;
  // return (size_t)t.millitm + t.time * 1000;
}
