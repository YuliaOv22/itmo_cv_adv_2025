from collections import defaultdict

class TrackingMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.data = []
        self.cb_id_to_tracks = defaultdict(list)
        self.track_id_to_cb = defaultdict(list)
        self.id_switches = defaultdict(int)
        self.prev_matches = {}
        self.total_detections = 0
    
    def add_frame(self, frame_data):
        """Добавляет данные фрейма для последующего расчета метрик"""
        frame_info = {
            'frame_id': frame_data['frame_id'],
            'objects': []
        }
        
        for obj in frame_data['data']:
            if 'cb_id' in obj and 'track_id' in obj:
                frame_info['objects'].append({
                    'cb_id': obj['cb_id'],
                    'track_id': obj['track_id'],
                    'has_bbox': bool(obj['bounding_box'])
                })
                
                self.cb_id_to_tracks[obj['cb_id']].append(obj['track_id'])
                self.track_id_to_cb[obj['track_id']].append(obj['cb_id'])
                
                if obj['bounding_box']:
                    self.total_detections += 1
                
                if obj['cb_id'] in self.prev_matches:
                    if self.prev_matches[obj['cb_id']] != obj['track_id']:
                        self.id_switches[obj['cb_id']] += 1
                self.prev_matches[obj['cb_id']] = obj['track_id']
        
        self.data.append(frame_info)
    
    def calculate_metrics(self):
        """Вычисляет все метрики трекинга"""
        metrics = {
            'id_precision': self.calculate_id_precision(),
            'id_recall': self.calculate_id_recall(),
            'avg_track_duration': self.calculate_avg_track_duration(),
            'id_switch_count': self.calculate_id_switch_count(),
            'fragmentation_count': self.calculate_fragmentation_count(),
            'mismatch_ratio': self.calculate_mismatch_ratio(),
            'avg_track_coverage': self.calculate_avg_track_coverage()
        }
        return metrics
    
    def calculate_id_precision(self):
        """Точность соответствия track_id реальному cb_id"""
        correct = 0
        total = 0
        
        for track_id, cb_ids in self.track_id_to_cb.items():
            if not cb_ids:
                continue
                
            most_common_cb = max(set(cb_ids), key=cb_ids.count)
            correct += cb_ids.count(most_common_cb)
            total += len(cb_ids)
        
        return correct / total if total > 0 else 0
    
    def calculate_id_recall(self):
        """Доля правильно отслеженных уникальных cb_id"""
        correctly_tracked = 0
        
        for cb_id, track_ids in self.cb_id_to_tracks.items():
            if not track_ids:
                continue
                
            most_common_track = max(set(track_ids), key=track_ids.count)
            if track_ids.count(most_common_track) / len(track_ids) > 0.5:
                correctly_tracked += 1
        
        return correctly_tracked / len(self.cb_id_to_tracks) if self.cb_id_to_tracks else 0
    
    def calculate_avg_track_duration(self):
        """Средняя продолжительность трека в кадрах"""
        track_durations = []
        
        for cb_id, track_ids in self.cb_id_to_tracks.items():
            if not track_ids:
                continue
                
            current_id = track_ids[0]
            current_length = 1
            
            for track_id in track_ids[1:]:
                if track_id == current_id:
                    current_length += 1
                else:
                    track_durations.append(current_length)
                    current_id = track_id
                    current_length = 1
            
            track_durations.append(current_length)
        
        return sum(track_durations) / len(track_durations) if track_durations else 0
    
    def calculate_id_switch_count(self):
        """Общее количество переключений ID"""
        return sum(self.id_switches.values())
    
    def calculate_fragmentation_count(self):
        """Количество разрывов треков (смена ID с возвратом к предыдущему)"""
        fragmentation = 0
        
        for cb_id, track_ids in self.cb_id_to_tracks.items():
            if len(track_ids) < 2:
                continue
                
            for i in range(1, len(track_ids)-1):
                if track_ids[i] != track_ids[i-1] and track_ids[i+1] == track_ids[i-1]:
                    fragmentation += 1
        
        return fragmentation
    
    def calculate_mismatch_ratio(self):
        """Коэффициент переключений ID = количество переключений / общее число детекций"""
        total_switches = sum(self.id_switches.values())
        return total_switches / self.total_detections if self.total_detections > 0 else 0
    
    def calculate_avg_track_coverage(self):
        """Средняя максимальная доля покрытия одним ID для всех треков"""
        coverages = []
        
        for cb_id, track_ids in self.cb_id_to_tracks.items():
            if not track_ids:
                continue
                
            track_counts = defaultdict(int)
            for tid in track_ids:
                track_counts[tid] += 1
            
            max_count = max(track_counts.values()) if track_counts else 0
            coverage = max_count / len(track_ids)
            coverages.append(coverage)
        
        return sum(coverages) / len(coverages) if coverages else 0
    
    def calculate_id_recall(self):
        """Доля правильно отслеженных уникальных cb_id"""
        correctly_tracked = 0
        
        for cb_id, track_ids in self.cb_id_to_tracks.items():
            if not track_ids:
                continue
                
            most_common_track = max(set(track_ids), key=track_ids.count)
            if track_ids.count(most_common_track) / len(track_ids) > 0.5:
                correctly_tracked += 1
        
        return correctly_tracked / len(self.cb_id_to_tracks) if self.cb_id_to_tracks else 0
    
    def report(self):
        """Выводит отчет по метрикам"""
        metrics = self.calculate_metrics()
        print("\nTracking Metrics Report:")
        print("=" * 40)
        print(f"ID Precision: {metrics['id_precision']:.3f}")
        print(f"ID Recall: {metrics['id_recall']:.3f}")
        print(f"Avg Track Duration: {metrics['avg_track_duration']:.1f} frames")
        print(f"ID Switch Count: {metrics['id_switch_count']}")
        print(f"Fragmentation Count: {metrics['fragmentation_count']}")
        print(f"Mismatch Ratio: {metrics['mismatch_ratio']:.4f}")
        print(f"Avg Track Coverage: {metrics['avg_track_coverage']:.3f}")
        print("=" * 40)