import random

host_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]

src_dst_pairs = []

for src in host_list:
    dst = random.choice(host_list)
    while dst == src:
        dst = random.choice(host_list)
    src_dst_pairs.append((src, dst))

print(src_dst_pairs)

base_rates = []
for i in range(len(src_dst_pairs)):
    base_rates.append(random.randint(1, 10))

print(base_rates)

fluctuation_amplitudes = []
for i in range(len(src_dst_pairs)):
    fluctuation_amplitudes.append(random.uniform(0, 1))

for i in range(len(fluctuation_amplitudes)):
    fluctuation_amplitudes[i] = round(fluctuation_amplitudes[i], 2)

print(fluctuation_amplitudes)


