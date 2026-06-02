const downloadButton = document.getElementById('downloadImage');
const downloadStatus = document.getElementById('actionStatus');

function setDownloadStatus(message, tone = 'info') {
    if (!downloadStatus) return;
    downloadStatus.textContent = message || '';
    downloadStatus.className = `action-status ${tone}`;
    downloadStatus.classList.toggle('is-visible', Boolean(message));
}

downloadButton?.addEventListener('click', async () => {
    const idleLabel = downloadButton.dataset.idleLabel || downloadButton.textContent.trim();
    downloadButton.dataset.idleLabel = idleLabel;
    downloadButton.disabled = true;
    downloadButton.textContent = '다운로드 준비 중...';
    setDownloadStatus('현재 active 모델의 예측 이미지를 내려받고 있습니다.', 'info');

    try {
        const response = await fetch('download');
        if (!response.ok) {
            throw new Error(`파일 다운로드 실패: ${response.status}`);
        }

        const blob = await response.blob();
        const blobUrl = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = blobUrl;
        link.download = 'melbourne_lstm_predictor.png';
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(blobUrl);
        setDownloadStatus('예측 이미지 다운로드가 완료되었습니다.', 'success');
    } catch (error) {
        console.error(error);
        setDownloadStatus(error.message || '다운로드 중 오류가 발생했습니다.', 'error');
    } finally {
        downloadButton.disabled = false;
        downloadButton.textContent = idleLabel;
    }
});
