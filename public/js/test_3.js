document.getElementById('downloadImage').addEventListener('click', function() {
    const url = "http://localhost:8080/view-download/"; // FastAPI 서버의 다운로드 엔드포인트

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('파일 다운로드 실패: ' + response.status);
            }
            return response.text(); // HTML 형식으로 응답 받기
        })
        .then(htmlContent => {
            // HTML 내용을 현재 페이지에 삽입하여 이미지를 표시
            document.body.innerHTML += htmlContent; // 기존 내용을 유지하면서 추가

            // 이미지 다운로드를 위한 Blob 처리
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = htmlContent;
            const imgElement = tempDiv.querySelector('img'); // <img> 태그 선택
            const imgSrc = imgElement.src; // 이미지 소스 가져오기

            return fetch(imgSrc); // 이미지 소스 URL로 Fetch
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('이미지 다운로드 실패: ' + response.status);
            }
            return response.blob(); // Blob 형태로 변환
        })
        .then(blob => {
            const blobUrl = window.URL.createObjectURL(blob); // Blob URL 생성
            const a = document.createElement('a'); // <a> 태그 생성
            a.style.display = 'none'; // 보이지 않도록 설정
            a.href = blobUrl;
            a.download = 'stock.png'; // 다운로드할 파일 이름
            document.body.appendChild(a); // DOM에 추가
            a.click(); // 클릭 이벤트 발생
            window.URL.revokeObjectURL(blobUrl); // 생성한 URL 해제
            
            // 다운로드 완료 메시지를 사용자에게 알림
            alert("파일이 성공적으로 다운로드되었습니다.");
        })
        .catch(error => {
            console.error(error);
            alert("다운로드 중 오류가 발생했습니다: " + error.message); // 오류 메시지 알림
        });
});
