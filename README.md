# 2023_FinAI_FinNewsGenerator
2023 대학원 금융인공지능 PBL : 공시 데이터 기반 금융 뉴스 생성
Test set은 샘플만 주어지고 비공개 된 데이터이다.

## 1. Methods
### 1. Filing only
- 공시 데이터만 encoder 입력으로 넣음.
  <img width="321" alt="image" src="https://github.com/na2na8/2023_FinAI_FinNewsGenerator/assets/32005272/e1535868-5585-431b-aade-5a2a3b45dcd7">
- 전처리는 개행문자 `\n`을 띄어쓰기 한 칸으로 대체함 ... `-`의 경우 숫자의 음수표현이나 내용이 없다는 것을 의미하기도 하므로 전처리 안 함
  
![image](https://github.com/na2na8/2023_FinAI_FinNewsGenerator/assets/32005272/34943bbe-e967-49d7-b1c7-45b90f52bb8b)

위와 같은 공시 데이터(표)는 아래의 json의 `filing_content`로 변환되어 있음

```json
[
  {
    "filing": {"id": 1032093, "dart_id": 20211208900413, "title": "타법인주식및출자증권취득결정", "date": 20211208, "time": 1647, "type_code": "I", "type_name": "거래소공시", "detail_type_code": "I001", "detail_type_name": "수시공시", "url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20211208900413"}, 
    "company": {"id": 6704, "dart_name": "유진기업", "stock_code": "023410", "market": "코스닥", "dart_code": 184667, "stock_name": "유진기업"}, 
    "file_path": "2021/12/20211208900413.html", 
    "filing_content": "유진기업/타법인주식및출자증권취득결정/(2021.12.08)타법인주식및출자증권취득결정\n타법인 주식 및 출자증권 취득결정\n1. 발행회사\n회사명(국적)\n유진더블유사모투자합자회사(대한민국)\n대표이사\n유진프라이빗에쿼티(주)\n자본금(원)\n-\n회사와 관계\n-\n발행주식총수(주)\n-\n주요사업\n금융업\n-최근 6월 이내 제3자 배정에 의한 신주취득 여부\n-\n2. 취득내역\n취득주식수(주)\n100,000,000,000\n취득금액(원)\n100,000,000,000\n자기자본(원)\n808,088,071,560\n자기자본대비(%)\n12.37\n대기업 여부\n해당\n3. 취득후 소유주식수 및 지분비율\n소유주식수(주)\n100,000,000,000\n지분비율(%)\n56.98\n4. 취득방법\n현금취득\n5. 취득목적\n사모투자합자회사 출자를 통한 배당 및 자본이득\n6. 취득예정일자\n2021-12-08\n7. 자산양수의 주요사항보고서 제출대상 여부\n아니오\n-최근 사업연도말 자산총액(원)\n4,708,657,424,261\n취득가액/자산총액(%)\n2.12\n8. 우회상장 해당 여부\n해당사항없음\n-향후 6월이내 제3자배정 증자 등 계획\n해당사항없음\n9. 발행회사(타법인)의 우회상장 요건 충족여부\n해당사항없음\n10. 이사회결의일(결정일)\n2021-12-08\n-사외이사 참석여부\n참석(명)\n1\n불참(명)\n-\n-감사(감사위원) 참석여부\n참석\n11. 공정거래위원회 신고대상 여부\n미해당\n12. 풋옵션계약 등의 체결여부\n아니오\n-계약내용\n-\n13. 기타 투자판단에 참고할 사항\n- 상기 2항 취득주식수와 3항 소유주식수는 좌수를 의미.\n- 발행회사는 신설법인으로 [발행회사의 요약 재무상황] 해당사항 없음.\n[발행회사의 요약 재무상황]\n(단위 : 백만원)\n구분\n자산총계\n부채총계\n자본총계\n자본금\n매출액\n당기순이익\n당해년도\n-\n-\n-\n-\n-\n-\n전년도\n-\n-\n-\n-\n-\n-\n전전년도\n-\n-\n-\n-\n-\n-"
  }
]
```

### 2. Keywords + Filing
- `filing_content`에서 키워드 생성  

  <img width="437" alt="image" src="https://github.com/na2na8/2023_FinAI_FinNewsGenerator/assets/32005272/afd778f6-c6bd-4abe-a4e2-7a27d4c6e027">

- 토크나이저에 스페셜 토큰인 `[KEYWORD]`, `[FILING]`추가하여 다음 두 가지 타입의 input으로 실험
  ```
  [KEYWORD] some keywords ... [FILING] some filing content ...
  [FILING] some filing content ... [KEYWORD] some keywords ...
  ```

### 3. Numbers + Filing    

<img width="395" alt="image" src="https://github.com/na2na8/2023_FinAI_FinNewsGenerator/assets/32005272/4f6df064-15a4-4d35-9a3b-f4de292afc6c">    

- `filing_content`에서 숫자 추출 (숫자, '.', ',', '-'로 구성된 문자열 추출)
  

- 토크나이저에 스페셜 토큰인 `[NUMBERS]`, `[FILING]` 추가
  ```
  [NUMBERS] some numbers ... [FILING] some filing content ...
  [FILING] some filing content ... [NUMBERS] some numbers ...
  ```

## 2. Hyperparams
- model : [SKT KoBART](https://github.com/SKT-AI/KoBART)
- batch size : 16
- max_length : 512
- learning_rate : 1e-5
- optimizer : AdamW

## 3. Analysis
### 1. 모델 정의
|Model|Input Style|
|---|---|
|Filing|filing content only|
|Keywords + Filing|`[KEYWORD]keywords[FILING]filing content`|
|Filing + Keywords|`[FILING]filing content[KEYWORD]keywords`|
|Numbers + Filing|`[NUMBERS]numbers[FILING]filing content`|
|Filing + Numbers|`[FILING]filing content[NUMBERS]numbers`|

### 2. Validation에서의 결과
|Model|ROUGE-1|ROUGE-2|ROUGE-L|
|:---:|:---:|:---:|:---:|
|_Filing_|0.565|0.402|0.559|
|Keywords + Filing|0.558|0.397|0.553|
|_Filing + Keywords_|0.561|0.398|0.556|
|Numbers + Filing|0.552|0.385|0.546|
|__Filing + Numbers__|0.565|0.404|0.559|

### 3. Validation 생성 결과
- 금액 관련 숫자 잘 생성하지 못함
- 상승률과 같은 퍼센트 수치, 날짜는 잘 생성함
  <img width="998" alt="image" src="https://github.com/na2na8/2023_FinAI_FinNewsGenerator/assets/32005272/0c84e117-12ee-41cd-8743-2b4a54bdfe7a">

### 4. `Filing + Numbers`모델의 Test 샘플에서의 생성 결과
<img width="812" alt="image" src="https://github.com/na2na8/2023_FinAI_FinNewsGenerator/assets/32005272/5e1216d7-31b6-4881-b893-140e35014808">

## 4. 결론
- 상위 3개 모델 결과 모두에서 금액 부분은 생성이 제대로 되지 않음을 확인
- Test Sample에서 Numbers 모델에서 유의미한 모습 보여줌
- 전체 Test Dataset에 대한 결과(교수님 채점 결과) : 전체 2등
  |ROUGE-1|ROUGE-2|ROUGE-L|
  |:---:|:---:|:---:|
  |0.6776|0.4427|0.6505|




