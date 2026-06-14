import csv
import sys
import time
import hashlib
import os

try:
    import pandas as pd
except ImportError:
    pass

# ==========================================
# 1. 스트리밍 데이터 제너레이터
# ==========================================
def stream_generator(file_path):
    """CSV 파일을 한 줄씩 읽어 반환하는 제너레이터"""
    with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

# ==========================================
# 2. Bloom Filter 구현
# ==========================================
class BloomFilter:
    def __init__(self, m, k):
        self.m = m 
        self.k = k 
        self.bit_array = bytearray((m + 7) // 8)
        
    def _get_hashes(self, item):
        hashes = []
        base_str = str(item).encode('utf-8')
        for i in range(self.k):
            h = int(hashlib.md5(base_str + str(i).encode('utf-8')).hexdigest(), 16)
            hashes.append(h % self.m)
        return hashes

    def add(self, item):
        for h in self._get_hashes(item):
            byte_idx = h // 8
            bit_idx = h % 8
            self.bit_array[byte_idx] |= (1 << bit_idx)

    def check(self, item):
        for h in self._get_hashes(item):
            byte_idx = h // 8
            bit_idx = h % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    def get_memory_size(self):
        return sys.getsizeof(self.bit_array)

# ==========================================
# 3. Count-Min Sketch 구현
# ==========================================
class CountMinSketch:
    def __init__(self, w, d):
        self.w = w  
        self.d = d  
        self.table = [[0] * w for _ in range(d)]

    def _get_hashes(self, item):
        hashes = []
        base_str = str(item).encode('utf-8')
        for i in range(self.d):
            h = int(hashlib.md5(base_str + str(i).encode('utf-8')).hexdigest(), 16)
            hashes.append(h % self.w)
        return hashes

    def add(self, item):
        hashes = self._get_hashes(item)
        for i, h in enumerate(hashes):
            self.table[i][h] += 1

    def estimate(self, item):
        hashes = self._get_hashes(item)
        return min(self.table[i][h] for i, h in enumerate(hashes))

    def get_memory_size(self):
        size = sys.getsizeof(self.table)
        for row in self.table:
            size += sys.getsizeof(row)
        return size

# ==========================================
# 4. 단일 실험 파이프라인 함수
# ==========================================
def run_single_experiment(exp_name, file_path, bf_m, bf_k, cms_w, cms_d):
    print(f"\n[{exp_name} 실험 시작] BF(m={bf_m}, k={bf_k}) / CMS(w={cms_w}, d={cms_d})")
    
    bf = BloomFilter(m=bf_m, k=bf_k)
    cms = CountMinSketch(w=cms_w, d=cms_d)
    
    gt_set = set()
    gt_dict = {}

    total_records = 0
    bf_false_positives = 0
    bf_true_negatives = 0

    start_time = time.time()

    for row in stream_generator(file_path):
        total_records += 1
        item = row.get('StockCode')
        if not item:
            continue

        is_in_gt = item in gt_set
        is_in_bf = bf.check(item)
        
        if not is_in_gt:
            bf_true_negatives += 1
            if is_in_bf:
                bf_false_positives += 1

        gt_set.add(item)
        gt_dict[item] = gt_dict.get(item, 0) + 1
        
        bf.add(item)
        cms.add(item)

    end_time = time.time()
    processing_time = end_time - start_time

    # 정확도 계산
    fpr = (bf_false_positives / bf_true_negatives) * 100 if bf_true_negatives > 0 else 0
    
    total_relative_error = 0
    for item, true_count in gt_dict.items():
        est_count = cms.estimate(item)
        total_relative_error += abs(est_count - true_count) / true_count
    avg_relative_error = (total_relative_error / len(gt_dict)) * 100

    # 결과 출력
    print(f"  ▶ [데이터] 처리 레코드: {total_records:,}건 | 고유 아이템(StockCode): {len(gt_dict):,}종")
    if processing_time > 0:
        print(f"  ▶ [시간] 총 처리 시간: {processing_time:.2f}초 | 초당 처리량: {total_records / processing_time:,.0f} records/sec")
    else:
        print(f"  ▶ [시간] 총 처리 시간: {processing_time:.2f}초 (너무 빨라 초당 처리량 계산 불가)")
        
    print(f"  ▶ [메모리] GT Set: {sys.getsizeof(gt_set):,} Bytes | Bloom Filter: {bf.get_memory_size():,} Bytes ({(bf.get_memory_size()/sys.getsizeof(gt_set))*100:.2f}%)")
    print(f"  ▶ [메모리] GT Dict: {sys.getsizeof(gt_dict):,} Bytes | CMS: {cms.get_memory_size():,} Bytes ({(cms.get_memory_size()/sys.getsizeof(gt_dict))*100:.2f}%)")
    
    print(f"  ▶ [정확도] Bloom Filter FPR: {fpr:.4f}%")
    print(f"  ▶ [정확도] Count-Min Sketch 평균 상대 오차: {avg_relative_error:.4f}%")
    print("-" * 60)

# ==========================================
# 5. 메인 실행부
# ==========================================
def run_all_experiments():
    excel_file_path = '/content/Online Retail.xlsx'
    csv_file_path = '/content/Online Retail.csv'
    
    print("=" * 60)
    print("데이터 파일 사전 검증 및 준비")
    print("=" * 60)

    # 1. 엑셀 파일인지 확인하고 CSV가 없으면 변환 (데이터 준비 과정)
    if os.path.exists(excel_file_path) and not os.path.exists(csv_file_path):
        print(f"[{excel_file_path}] 파일이 감지되었습니다.")
        print("스트리밍 처리를 위해 CSV 포맷으로 변환을 시작합니다. (데이터 용량에 따라 1~2분 소요될 수 있습니다...)")
        try:
            df = pd.read_excel(excel_file_path)
            df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
            print(f"✅ CSV 변환 완료: {csv_file_path}")
        except Exception as e:
            print(f"❌ 엑셀 변환 중 오류 발생: {e}")
            return
    elif not os.path.exists(excel_file_path) and not os.path.exists(csv_file_path):
        print("❌ 파일을 찾을 수 없습니다. 좌측 폴더 탭에 '/content/Online Retail.xlsx' 파일이 업로드되어 있는지 확인해주세요.")
        return
    else:
        print("✅ 사용 가능한 CSV 파일이 이미 존재합니다.")

    print("\n" + "=" * 60)
    print("스트리밍 알고리즘 트레이드오프 분석 실험 시작")
    print(f"스트리밍 대상 파일: {csv_file_path}")
    print("=" * 60)

    # 3단계 실험 세팅
    experiments = [
        {"name": "Exp 1 (최소 파라미터)", "bf_m": 100_000, "bf_k": 3, "cms_w": 1_000, "cms_d": 3},
        {"name": "Exp 2 (중간 파라미터)", "bf_m": 500_000, "bf_k": 5, "cms_w": 5_000, "cms_d": 5},
        {"name": "Exp 3 (최대 파라미터)", "bf_m": 1_000_000, "bf_k": 7, "cms_w": 10_000, "cms_d": 7},
    ]

    for exp in experiments:
        run_single_experiment(
            exp_name=exp["name"], 
            file_path=csv_file_path,
            bf_m=exp["bf_m"], 
            bf_k=exp["bf_k"], 
            cms_w=exp["cms_w"], 
            cms_d=exp["cms_d"]
        )
    
    print("\n🎉 모든 실험이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    run_all_experiments()
