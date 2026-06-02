const API_ROOT = ''; // Using relative paths to work with FastAPI's root_path/base href

const uploadForm = document.getElementById('uploadForm');
const uploadSubmitBtn = document.getElementById('uploadSubmitBtn');
const fileUploadInput = document.getElementById('fileUpload');
const selectedFileName = document.getElementById('selectedFileName');
const fileName = document.getElementById('fileName');
const resultSummary = document.getElementById('resultSummary');
const imageContainer = document.getElementById('imageContainer');
const resultMetrics = document.getElementById('resultMetrics');
const actionStatus = document.getElementById('actionStatus');
const approvalModalEl = document.getElementById('approvalModal');
const approvalModal = approvalModalEl ? new bootstrap.Modal(approvalModalEl) : null;
const approvalResultModalEl = document.getElementById('approvalResultModal');
const approvalResultModal = approvalResultModalEl ? new bootstrap.Modal(approvalResultModalEl) : null;
const approvalReportTitle = document.getElementById('approvalReportTitle');
const approvalReportSummary = document.getElementById('approvalReportSummary');
const approvalReportDetails = document.getElementById('approvalReportDetails');
const approvalComparisonBanner = document.getElementById('approvalComparisonBanner');
const approvalComparisonCards = document.getElementById('approvalComparisonCards');
const approvalComparisonCharts = document.getElementById('approvalComparisonCharts');
const approvalRecommendationBox = document.getElementById('approvalRecommendationBox');
const approvalRecommendationTitle = document.getElementById('approvalRecommendationTitle');
const approvalRecommendationBody = document.getElementById('approvalRecommendationBody');
const approveRetrainBtn = document.getElementById('approveRetrainBtn');
const rejectRetrainBtn = document.getElementById('rejectRetrainBtn');
const approvalOutcomeBanner = document.getElementById('approvalOutcomeBanner');
const approvalOutcomeCards = document.getElementById('approvalOutcomeCards');
const approvalOutcomeCharts = document.getElementById('approvalOutcomeCharts');
const approvalOutcomeSeverityBadge = document.getElementById('approvalOutcomeSeverityBadge');
const approvalOutcomeTitle = document.getElementById('approvalOutcomeTitle');
const approvalOutcomeSubtitle = document.getElementById('approvalOutcomeSubtitle');
const approvalOutcomeBeforeMape = document.getElementById('approvalOutcomeBeforeMape');
const approvalOutcomeAfterMape = document.getElementById('approvalOutcomeAfterMape');
const approvalOutcomeBeforeRmse = document.getElementById('approvalOutcomeBeforeRmse');
const approvalOutcomeAfterRmse = document.getElementById('approvalOutcomeAfterRmse');
const approvalOutcomeRecommendationBox = document.getElementById('approvalOutcomeRecommendationBox');
const approvalOutcomeRecommendationTitle = document.getElementById('approvalOutcomeRecommendationTitle');
const approvalOutcomeRecommendationBody = document.getElementById('approvalOutcomeRecommendationBody');
const approvalOutcomeDetails = document.getElementById('approvalOutcomeDetails');
const closeApprovalResultBtn = document.getElementById('closeApprovalResultBtn');
const overviewTarget = document.getElementById('overviewTarget');
const overviewModel = document.getElementById('overviewModel');
const overviewThresholdRmse = document.getElementById('overviewThresholdRmse');
const overviewStatus = document.getElementById('overviewStatus');
const overviewActiveMape = document.getElementById('overviewActiveMape');
const overviewActiveRmse = document.getElementById('overviewActiveRmse');
const overviewBaselineMape = document.getElementById('overviewBaselineMape');
const overviewBaselineRmse = document.getElementById('overviewBaselineRmse');
const overviewThresholdCard = document.getElementById('overviewThresholdCard');
const overviewExceedRatio = document.getElementById('overviewExceedRatio');
const overviewModelFigure = document.getElementById('overviewModelFigure');
const initialStreamSummary = document.getElementById('initialStreamSummary');
const initialChartContainer = document.getElementById('initialChartContainer');
const retrainingDecisionPanel = document.getElementById('retrainingDecisionPanel');
const decisionPanelTitle = document.getElementById('decisionPanelTitle');
const decisionPanelBody = document.getElementById('decisionPanelBody');
const decisionPanelMetrics = document.getElementById('decisionPanelMetrics');
const decisionPanelActions = document.getElementById('decisionPanelActions');
const inlineKeepModelBtn = document.getElementById('inlineKeepModelBtn');
const inlineRetrainBtn = document.getElementById('inlineRetrainBtn');

let pendingSavedFilename = null;
let pendingComparisonPreview = null;
let approvalPopupWindow = null;

function cleanupModalArtifacts() {
    if (document.querySelector('.modal.show')) {
        return;
    }
    document.querySelectorAll('.modal-backdrop').forEach((backdrop) => backdrop.remove());
    document.body.classList.remove('modal-open');
    document.body.style.removeProperty('padding-right');
    document.body.style.removeProperty('overflow');
}

function formatMetric(value, digits = 3) {
    return Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
}

function formatPercent(value, digits = 2) {
    return Number.isFinite(value) ? `${value.toFixed(digits)}%` : 'N/A';
}

function readButtonLabel(button) {
    if (!button) return '';
    return button.tagName === 'INPUT' ? button.value : button.textContent.trim();
}

function writeButtonLabel(button, label) {
    if (!button) return;
    if (button.tagName === 'INPUT') {
        button.value = label;
        return;
    }
    button.textContent = label;
}

function setButtonBusy(button, isBusy, busyLabel) {
    if (!button) return;
    if (!button.dataset.idleLabel) {
        button.dataset.idleLabel = readButtonLabel(button);
    }
    button.disabled = isBusy;
    button.classList.toggle('is-busy', isBusy);
    writeButtonLabel(button, isBusy ? busyLabel : button.dataset.idleLabel);
}

function setActionStatus(message, tone = 'info') {
    if (!actionStatus) return;
    actionStatus.textContent = message || '';
    actionStatus.className = `action-status ${tone}`;
    actionStatus.classList.toggle('is-visible', Boolean(message));
}

function hideRetrainingDecisionPanel() {
    if (!retrainingDecisionPanel) return;
    retrainingDecisionPanel.style.display = 'none';
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function metricValue(metrics, key) {
    const value = metrics?.forecast?.[key];
    return Number.isFinite(value) ? value : null;
}

function pillToneForSeverity(severity) {
    if (severity === 'high') return 'warning';
    if (severity === 'medium') return 'info';
    return 'success';
}

function buildPill(label, tone = 'neutral') {
    return `<span class="status-pill ${tone}">${escapeHtml(label)}</span>`;
}

function buildMetricCard(title, value, helper) {
    return `
        <div class="col-md-6 col-xl-3">
            <div class="metric-card">
                <div class="label">${escapeHtml(title)}</div>
                <div class="value">${escapeHtml(value)}</div>
                <div class="helper">${escapeHtml(helper)}</div>
            </div>
        </div>
    `;
}

function buildOutcomeCard(title, value, description, accentColor) {
    return `
        <div class="col-md-4">
            <div class="metric-card" style="border-top: 4px solid ${accentColor};">
                <div class="label" style="color: ${accentColor};">${escapeHtml(title)}</div>
                <div class="value">${escapeHtml(value)}</div>
                <div class="helper">${escapeHtml(description)}</div>
            </div>
        </div>
    `;
}

function buildRetrainMetricCard(title, mape, rmse, helper, tone = 'active') {
    const cardClass = tone === 'candidate' ? 'candidate' : 'active';
    return `
        <div class="comparison-card ${cardClass}">
            <div class="comparison-card__eyebrow">${tone === 'candidate' ? '재학습 후보 모델' : '현재 운영 모델'}</div>
            <div class="comparison-card__title">${escapeHtml(title)}</div>
            <div class="comparison-card__meta">${escapeHtml(helper)}</div>
            <div class="comparison-card__stats">
                <div class="comparison-card__stat">
                    <div class="comparison-card__stat-label">MAPE</div>
                    <div class="comparison-card__stat-value">${escapeHtml(formatPercent(mape))}</div>
                </div>
                <div class="comparison-card__stat">
                    <div class="comparison-card__stat-label">RMSE</div>
                    <div class="comparison-card__stat-value">${escapeHtml(formatMetric(rmse))}</div>
                </div>
            </div>
        </div>
    `;
}

function buildComparisonCard(kind, title, meta, metrics) {
    const cardClass = kind === 'candidate' ? 'candidate' : 'active';
    return `
        <div class="comparison-card ${cardClass}">
            <div class="comparison-card__eyebrow">${kind === 'candidate' ? '변경 후 모델' : '변경 전 모델'}</div>
            <div class="comparison-card__title">${escapeHtml(title)}</div>
            <div class="comparison-card__meta">${escapeHtml(meta)}</div>
            <div class="comparison-card__stats">
                <div class="comparison-card__stat">
                    <div class="comparison-card__stat-label">비교 구간 MAPE</div>
                    <div class="comparison-card__stat-value">${escapeHtml(formatPercent(metrics?.mape))}</div>
                </div>
                <div class="comparison-card__stat">
                    <div class="comparison-card__stat-label">비교 구간 RMSE</div>
                    <div class="comparison-card__stat-value">${escapeHtml(formatMetric(metrics?.rmse))}</div>
                </div>
            </div>
        </div>
    `;
}

function buildComparisonChartCard(title, caption, src) {
    return `
        <div class="comparison-chart-card">
            <div class="comparison-chart-card__title">${escapeHtml(title)}</div>
            <div class="comparison-chart-card__copy">${escapeHtml(caption)}</div>
            <div class="comparison-chart-card__frame">
                <img src="${src}" alt="${escapeHtml(title)}" />
            </div>
        </div>
    `;
}

function hideLegacyApprovalOutcomeBlocks() {
    if (approvalOutcomeCards) {
        approvalOutcomeCards.style.display = 'none';
        approvalOutcomeCards.innerHTML = '';
    }

    if (approvalOutcomeDetails) {
        approvalOutcomeDetails.innerHTML = '';
        approvalOutcomeDetails.style.display = 'none';
        const detailSection = approvalOutcomeDetails.closest('.report-section');
        if (detailSection) {
            detailSection.style.display = 'none';
        }
    }
}

function buildModelCard(title, subtitle, metrics, tone, footerNote) {
    const mape = formatPercent(metricValue(metrics, 'mape'));
    const rmse = formatMetric(metricValue(metrics, 'rmse'));
    const mae = formatMetric(metricValue(metrics, 'mae'));
    const tonePill = tone === 'active'
        ? buildPill('현재 서빙 모델', 'success')
        : buildPill('비교용 베이스라인', 'info');

    return `
        <div class="col-lg-6">
            <div class="model-card">
                <div class="title-row">
                    <div>
                        <h4>${escapeHtml(title)}</h4>
                        <div class="subtitle">${escapeHtml(subtitle)}</div>
                    </div>
                    ${tonePill}
                </div>
                <div class="metric-grid">
                    <div class="cell">
                        <div class="cell-label">MAPE</div>
                        <div class="cell-value">${escapeHtml(mape)}</div>
                    </div>
                    <div class="cell">
                        <div class="cell-label">RMSE</div>
                        <div class="cell-value">${escapeHtml(rmse)}</div>
                    </div>
                    <div class="cell">
                        <div class="cell-label">MAE</div>
                        <div class="cell-value">${escapeHtml(mae)}</div>
                    </div>
                </div>
                <div class="footer-note">${escapeHtml(footerNote)}</div>
            </div>
        </div>
    `;
}

function buildReportMetricCard(title, value, helper) {
    return `
        <div class="col-sm-6 col-xl-3">
            <div class="metric-card">
                <div class="label">${escapeHtml(title)}</div>
                <div class="value">${escapeHtml(value)}</div>
                <div class="helper">${escapeHtml(helper)}</div>
            </div>
        </div>
    `;
}

function buildImageCard(title, src, caption) {
    return `
        <div class="col-lg-6">
            <div class="model-card">
                <div class="title-row">
                    <div>
                        <h4>${escapeHtml(title)}</h4>
                        <div class="subtitle">${escapeHtml(caption)}</div>
                    </div>
                </div>
                <div style="border-radius: 16px; overflow: hidden; border: 1px solid #dbe5f0; background: #f8fbff;">
                    <img src="${src}" alt="${escapeHtml(title)}" style="display:block; width:100%; height:auto;" />
                </div>
            </div>
        </div>
    `;
}

function buildDecisionMetric(label, value) {
    return `
        <div class="decision-metric">
            <div class="decision-metric__label">${escapeHtml(label)}</div>
            <div class="decision-metric__value">${escapeHtml(value)}</div>
        </div>
    `;
}

function renderOverviewModelFigure(title, src, caption) {
    if (!overviewModelFigure) return;
    if (!src) {
        overviewModelFigure.innerHTML = `
            <div class="report-figure-block__title">${escapeHtml(title || 'Current Operating Forecast')}</div>
            <div class="report-figure-block__caption">${escapeHtml(caption || '표시할 그래프가 아직 없습니다.')}</div>
        `;
        return;
    }
    overviewModelFigure.innerHTML = `
        <div class="report-figure-block__title">${escapeHtml(title)}</div>
        <div class="report-figure-block__caption">${escapeHtml(caption)}</div>
        <div class="report-figure-block__frame">
            <img src="${src}" alt="${escapeHtml(title)}" />
        </div>
    `;
}

function describeApprovalStatus(data, report, activeForecast) {
    if (data.approval_required) {
        return `현재 업로드 구간은 재학습 검토 대상입니다. MAPE ${formatPercent(activeForecast?.mape)}를 기록했고, 승인 시 여러 후보 조합을 재탐색합니다.`;
    }
    if (report?.status_label) {
        return `${report.status_label}. 현재 active 모델이 임계 범위 안에서 동작하고 있어 즉시 재학습은 필요하지 않습니다.`;
    }
    return '현재 active 모델이 정상 범위 안에서 동작하고 있습니다.';
}

function updateSelectedFileName(name) {
    if (!selectedFileName) return;
    selectedFileName.textContent = name || '선택된 파일 없음';
}

function renderOverviewMetrics({
    targetColumn,
    modelFamily,
    thresholdRmse,
    activeMape,
    activeRmse,
    baselineMape,
    baselineRmse,
    exceedPercent,
    approvalRequired,
    statusLabel,
}) {
    if (overviewTarget) {
        overviewTarget.textContent = targetColumn === 'Chemical Oxygen Demand' ? 'COD' : (targetColumn || 'N/A');
    }
    if (overviewModel) {
        const labelMap = {
            random_forest: 'RandomForest',
            gru: 'GRU',
            lstm: 'LSTM',
            conv1d: 'Conv1D',
            hist_gb: 'HistGB',
            extra_trees: 'ExtraTrees',
        };
        overviewModel.textContent = labelMap[modelFamily] || modelFamily || 'N/A';
    }
    if (overviewThresholdRmse) {
        overviewThresholdRmse.textContent = formatMetric(thresholdRmse, 2);
    }
    if (overviewThresholdCard) {
        overviewThresholdCard.textContent = formatMetric(thresholdRmse, 2);
    }
    if (overviewActiveMape) {
        overviewActiveMape.textContent = formatPercent(activeMape);
    }
    if (overviewActiveRmse) {
        overviewActiveRmse.textContent = formatMetric(activeRmse, 2);
    }
    if (overviewBaselineMape) {
        overviewBaselineMape.textContent = Number.isFinite(baselineMape) ? formatPercent(baselineMape) : 'N/A';
    }
    if (overviewBaselineRmse) {
        overviewBaselineRmse.textContent = formatMetric(baselineRmse, 2);
    }
    if (overviewExceedRatio) {
        overviewExceedRatio.textContent = Number.isFinite(exceedPercent) ? formatPercent(exceedPercent) : 'N/A';
        overviewExceedRatio.style.color = Number.isFinite(exceedPercent) && exceedPercent > 0 ? '#c2410c' : '#15803d';
    }
    if (overviewStatus) {
        overviewStatus.textContent = statusLabel || (approvalRequired ? '재학습 검토' : '안정적 운영');
    }
}

function renderCurrentModelReport({
    savedFilename,
    targetColumn,
    modelName,
    thresholdRmse,
    mape,
    rmse,
    mae,
    statusLabel,
    approvalRequired,
    timestamp,
}) {
    if (!resultMetrics) return;

    resultMetrics.innerHTML = `
        <div class="col-12">
            <div class="report-document">
                    <div class="report-header">
                        <div class="title">현재 모델 평가 보고서</div>
                        <div class="meta">
                        <span>파일명: <strong>${escapeHtml(savedFilename || 'uploaded.csv')}</strong></span>
                        <span>생성일시: ${timestamp}</span>
                        <span>예측대상: ${escapeHtml(targetColumn)}</span>
                        </div>
                    </div>

                <div class="metric-summary-grid">
                    <div class="metric-summary-item">
                        <div class="label">현재 MAPE</div>
                        <div class="value">${formatPercent(mape)}</div>
                    </div>
                    <div class="metric-summary-item">
                        <div class="label">현재 RMSE</div>
                        <div class="value">${formatMetric(rmse)}</div>
                    </div>
                    <div class="metric-summary-item">
                        <div class="label">기준 RMSE</div>
                        <div class="value">${formatMetric(thresholdRmse)}</div>
                    </div>
                    <div class="metric-summary-item">
                        <div class="label">판정</div>
                        <div class="value">${escapeHtml(statusLabel)}</div>
                    </div>
                </div>

                <div class="report-section-title">현재 모델 평가</div>

                <div class="report-table-container">
                    <table class="report-table">
                        <thead>
                            <tr>
                                <th>구분</th>
                                <th>모델 / 기준</th>
                                <th>MAPE</th>
                                <th>RMSE</th>
                                <th>MAE</th>
                                <th>판정</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>현재 운영 모델</strong></td>
                                <td>
                                    ${escapeHtml(modelName || 'melbourne_deployed_model')}<br>
                                    <small class="text-muted">기준 RMSE ${formatMetric(thresholdRmse)}</small>
                                </td>
                                <td class="fw-bold">${formatPercent(mape)}</td>
                                <td>${formatMetric(rmse, 2)}</td>
                                <td>${formatMetric(mae, 2)}</td>
                                <td>${buildPill(statusLabel, approvalRequired ? 'warning' : 'success')}</td>
                            </tr>
                            <tr>
                                <td><strong>기준선</strong></td>
                                <td>
                                    RMSE 기준선<br>
                                    <small class="text-muted">validation / calibration 기반</small>
                                </td>
                                <td>N/A</td>
                                <td>${formatMetric(thresholdRmse, 2)}</td>
                                <td>N/A</td>
                                <td>${buildPill(approvalRequired ? '초과' : '정상', approvalRequired ? 'warning' : 'success')}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="report-section-title">요약 코멘트</div>

                <div class="report-insight-box">
                    <div class="title">요약 의견</div>
                    <p class="content">
                        ${escapeHtml(
                            approvalRequired
                                ? `현재 평가 구간에서 RMSE ${formatMetric(rmse)}가 기준 ${formatMetric(thresholdRmse)}를 넘어 추가 검토가 필요합니다.`
                                : `현재 평가 구간에서 RMSE ${formatMetric(rmse)}로 기준 ${formatMetric(thresholdRmse)} 이내에 있어 현재 운영 모델을 유지할 수 있습니다.`
                        )}
                    </p>
                </div>
            </div>
        </div>
    `;
}

function renderInitialDashboard(data) {
    const overview = data?.overview || {};
    renderOverviewMetrics({
        targetColumn: overview.target_column,
        modelFamily: overview.model_family,
        thresholdRmse: overview.threshold_rmse,
        activeMape: overview.calibration_mape ?? overview.active_mape,
        activeRmse: overview.calibration_rmse,
        baselineMape: overview.baseline_mape,
        baselineRmse: overview.baseline_rmse,
        exceedPercent: null,
        approvalRequired: false,
        statusLabel: overview.status_label || '기준선 로드됨',
    });

    if (initialStreamSummary) {
        initialStreamSummary.textContent = data?.summary_text || '기준 시계열을 불러왔습니다.';
    }

    if (Array.isArray(data?.charts) && data.charts.length > 0) {
        const currentModelChart = data.charts.find((chart) => /calibration forecast/i.test(chart.title || '')) || data.charts[0];
        renderOverviewModelFigure(
            currentModelChart?.title || 'Current Operating Forecast',
            currentModelChart?.src,
            currentModelChart?.caption || '현재 운영 모델 기준 대표 예측 그래프입니다.'
        );
    }

    if (initialChartContainer && Array.isArray(data?.charts) && data.charts.length > 0) {
        const remainingCharts = data.charts.filter((chart) => !/calibration forecast/i.test(chart.title || ''));
        initialChartContainer.innerHTML = remainingCharts.map((chart) => (
            buildImageCard(chart.title, chart.src, chart.caption)
        )).join('');
    }

    if (resultSummary && (!fileName || fileName.style.display === 'none')) {
        resultSummary.textContent = data?.summary_text || '기준 시계열과 보정 구간 예측을 먼저 확인한 뒤, 업로드로 운영 상황을 점검할 수 있습니다.';
    }
    hideRetrainingDecisionPanel();
}

function renderRetrainingDecisionPanel({
    approvalRequired = false,
    currentRmse = null,
    currentMape = null,
    thresholdRmse = null,
    exceedPercent = null,
    lockedMessage = '',
}) {
    if (!retrainingDecisionPanel) return;

    retrainingDecisionPanel.style.display = 'block';
    retrainingDecisionPanel.className = `summary-card decision-panel ${approvalRequired ? 'needs-action' : 'stable'}`;

    if (decisionPanelTitle) {
        decisionPanelTitle.textContent = lockedMessage
            ? '현재 모델 유지'
            : (approvalRequired ? '재학습 필요' : '현재 모델 유지 가능');
    }

    if (decisionPanelBody) {
        if (lockedMessage) {
            decisionPanelBody.textContent = lockedMessage;
        } else if (approvalRequired) {
            decisionPanelBody.textContent = '현재 모델의 RMSE가 기준선을 초과했습니다. 아래에서 현재 모델을 유지하거나 재학습을 실행할 수 있습니다.';
        } else {
            decisionPanelBody.textContent = '현재 모델의 RMSE가 기준선 이내에 있어 이번 배치에서는 재학습 없이 운영할 수 있습니다.';
        }
    }

    if (decisionPanelMetrics) {
        decisionPanelMetrics.innerHTML = [
            buildDecisionMetric('현재 MAPE', formatPercent(currentMape)),
            buildDecisionMetric('현재 RMSE', formatMetric(currentRmse)),
            buildDecisionMetric('기준 RMSE', formatMetric(thresholdRmse)),
            buildDecisionMetric('초과율', Number.isFinite(exceedPercent) ? formatPercent(exceedPercent) : 'N/A'),
        ].join('');
    }

    if (decisionPanelActions) {
        decisionPanelActions.style.display = (!lockedMessage && approvalRequired) ? 'flex' : 'none';
    }
}

function closeApprovalPopupIfOpen() {
    if (approvalPopupWindow && !approvalPopupWindow.closed) {
        approvalPopupWindow.close();
    }
    approvalPopupWindow = null;
}

function buildApprovalPopupHtml(report, comparisonPreview = null) {
    const currentRmse = comparisonPreview?.current_upload_metrics?.rmse ?? report?.metrics?.current_rmse;
    const thresholdRmse = report?.metrics?.threshold_rmse;
    const details = report?.findings || report?.details || [];

    return `
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>재학습 검토 보고서</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {
      margin: 0;
      background: #f3f4f6;
      color: #111827;
      font-family: "Segoe UI", Roboto, sans-serif;
    }
    .wrap {
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px;
    }
    .sheet {
      background: #ffffff;
      border: 1px solid #d1d5db;
      padding: 28px 32px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 1.8rem;
      line-height: 1.3;
    }
    .sub {
      color: #4b5563;
      line-height: 1.8;
      margin-bottom: 22px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px 28px;
      border-top: 1px solid #d1d5db;
      padding-top: 18px;
      margin-bottom: 24px;
    }
    .metric {
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 14px;
    }
    .metric .label {
      color: #6b7280;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .metric .value {
      font-size: 1.2rem;
      font-weight: 800;
    }
    .banner {
      border-top: 1px solid #d1d5db;
      border-bottom: 1px solid #d1d5db;
      padding: 16px 0;
      margin-bottom: 24px;
      color: #9a3412;
      font-weight: 700;
    }
    .section-title {
      font-size: 1rem;
      font-weight: 800;
      margin: 0 0 12px;
    }
    .summary-box {
      border-top: 1px solid #d1d5db;
      padding-top: 14px;
      margin-bottom: 24px;
      color: #4b5563;
      line-height: 1.8;
    }
    ul {
      margin: 0;
      padding-left: 18px;
      color: #4b5563;
      line-height: 1.8;
    }
    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 28px;
      padding-top: 18px;
      border-top: 1px solid #d1d5db;
    }
    button {
      min-height: 40px;
      padding: 0 16px;
      border: 1px solid #9ca3af;
      background: #ffffff;
      color: #111827;
      font-size: 0.9rem;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary {
      background: #111827;
      color: #ffffff;
      border-color: #111827;
    }
    @media (max-width: 720px) {
      .wrap { padding: 16px; }
      .sheet { padding: 20px; }
      .metrics { grid-template-columns: 1fr; }
      .actions { flex-direction: column; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="sheet">
      <h1>재학습 검토 보고서</h1>
      <div class="sub">${escapeHtml(report?.summary || '현재 업로드 데이터의 RMSE가 기준을 초과해 재학습 검토가 필요합니다.')}</div>

      <div class="metrics">
        <div class="metric">
          <div class="label">현재 RMSE</div>
          <div class="value">${escapeHtml(formatMetric(currentRmse))}</div>
        </div>
        <div class="metric">
          <div class="label">기준 RMSE</div>
          <div class="value">${escapeHtml(formatMetric(thresholdRmse))}</div>
        </div>
      </div>

      <div class="banner">현재 모델의 RMSE가 기준선을 초과했습니다. 새 후보를 재학습할지 결정해주세요.</div>

      <div class="section-title">요약</div>
      <div class="summary-box">
        현재 업로드 데이터에 대해 운영 모델 성능을 다시 평가한 결과, 재학습 검토 조건을 충족했습니다.
        현재 모델을 유지하거나 재학습을 실행해 후보 모델 개선 여부를 확인할 수 있습니다.
      </div>

      ${details.length ? `
        <div class="section-title">세부 내용</div>
        <ul>
          ${details.map((detail) => `<li>${escapeHtml(detail)}</li>`).join('')}
        </ul>
      ` : ''}

      <div class="actions">
        <button type="button" onclick="window.opener.handleApprovalPopupDecision(false); window.close();">현재 모델 유지</button>
        <button type="button" class="primary" onclick="window.opener.handleApprovalPopupDecision(true); window.close();">재학습 실행</button>
      </div>
    </div>
  </div>
</body>
</html>
    `;
}

function openApprovalPopup(report, comparisonPreview = null) {
    const popup = window.open('', 'melbourneApprovalPopup', 'width=1120,height=860,resizable=yes,scrollbars=yes');
    if (!popup) {
        return false;
    }
    approvalPopupWindow = popup;
    popup.document.open();
    popup.document.write(buildApprovalPopupHtml(report, comparisonPreview));
    popup.document.close();
    popup.focus();
    return true;
}

window.handleApprovalPopupDecision = function handleApprovalPopupDecision(approved) {
    submitApproval(approved);
};

function renderUploadResult(data) {
    const activeModel = data.model_1 || {};
    const activeForecast = activeModel.metrics?.forecast || {};
    const baselineModel = data.model_2 || null;
    const report = data.llm_report || activeModel.llm_report || {};
    const currentMape = metricValue(activeModel.metrics, 'mape');
    const thresholdRmse = Number.isFinite(activeModel.rmse_threshold)
        ? activeModel.rmse_threshold
        : report?.metrics?.threshold_rmse;
    const currentRmse = metricValue(activeModel.metrics, 'rmse');
    const exceedPercent = report?.metrics?.exceed_percent ?? (
        Number.isFinite(currentRmse) && Number.isFinite(thresholdRmse) && thresholdRmse > 0
            ? ((currentRmse / thresholdRmse) - 1) * 100
            : null
    );
    const targetColumn = activeModel.target_column || report?.target_column || 'Chemical Oxygen Demand';
    const timestamp = new Date().toLocaleString();
    const statusLabel = data.approval_required ? '재학습 필요' : '현재 모델 유지 가능';

    renderOverviewMetrics({
        targetColumn,
        modelFamily: activeModel.model_family,
        thresholdRmse,
        activeMape: currentMape,
        activeRmse: currentRmse,
        baselineMape: metricValue(baselineModel?.metrics, 'mape'),
        baselineRmse: metricValue(baselineModel?.metrics, 'rmse'),
        exceedPercent,
        approvalRequired: Boolean(data.approval_required),
        statusLabel,
    });
    hideRetrainingDecisionPanel();
    renderOverviewModelFigure(
        '현재 운영 모델 예측 그래프',
        data.result_visualizing_LSTM,
        `현재 active 모델(${activeModel.model_name || 'melbourne_deployed_model'})의 최신 예측 결과입니다.`
    );

    fileName.style.display = 'block';
    fileName.innerHTML = `<strong>${escapeHtml(data.saved_filename || 'uploaded.csv')}</strong>`;
    renderCurrentModelReport({
        savedFilename: data.saved_filename,
        targetColumn,
        modelName: activeModel.model_name || 'melbourne_deployed_model',
        thresholdRmse,
        mape: metricValue(activeModel.metrics, 'mape'),
        rmse: metricValue(activeModel.metrics, 'rmse'),
        mae: metricValue(activeModel.metrics, 'mae'),
        statusLabel,
        approvalRequired: Boolean(data.approval_required),
        timestamp,
    });

    resultSummary.textContent = data.approval_required
        ? `현재 업로드 배치는 기준 RMSE를 초과했습니다. 현재 모델 보고서를 확인한 뒤 재학습 실행 여부를 선택할 수 있습니다.`
        : `현재 업로드 배치는 기준선 이내로 평가되어 active 모델을 그대로 유지합니다.`;

    if (imageContainer) {
        imageContainer.innerHTML = '';
        if (data.result_visualizing_LSTM) {
            imageContainer.insertAdjacentHTML('beforeend', buildImageCard(
                '현재 모델 예측 그래프',
                data.result_visualizing_LSTM,
                `현재 운영 모델(${activeModel.model_name || 'melbourne_deployed_model'})의 실제값 대비 예측 그래프입니다.`
            ));
        }
    }
}

function renderApprovalReport(report, comparisonPreview = null) {
    const badge = document.getElementById('reportSeverityBadge');
    if (badge) {
        const severity = (report?.severity || 'medium').toLowerCase();
        badge.textContent = severity.toUpperCase();
        badge.className = `toss-badge ${severity}`;
    }

    if (approvalReportTitle) {
        approvalReportTitle.textContent = '재학습 필요';
    }
    if (approvalReportSummary) {
        approvalReportSummary.textContent = report?.summary || '현재 업로드 데이터의 RMSE가 기준을 초과해 재학습이 필요합니다.';
    }

    const modalCurrentRmse = document.getElementById('modalCurrentRmse');
    const modalThresholdRmse = document.getElementById('modalThresholdRmse');
    if (modalCurrentRmse) {
        modalCurrentRmse.textContent = formatMetric(
            comparisonPreview?.current_upload_metrics?.rmse ?? report?.metrics?.current_rmse
        );
    }
    if (modalThresholdRmse) {
        modalThresholdRmse.textContent = formatMetric(report?.metrics?.threshold_rmse);
    }

    if (approvalComparisonBanner) {
        approvalComparisonBanner.style.display = 'block';
        approvalComparisonBanner.style.background = '#fff7ed';
        approvalComparisonBanner.style.color = '#9a3412';
        approvalComparisonBanner.textContent = '현재 모델의 RMSE가 기준선을 초과했습니다. 재학습을 실행하면 후보 모델을 다시 학습하고 결과 보고서를 보여줍니다.';
    }

    if (approvalComparisonCards) {
        approvalComparisonCards.style.display = 'none';
        approvalComparisonCards.innerHTML = '';
    }

    if (approvalComparisonCharts) {
        approvalComparisonCharts.style.display = 'none';
        approvalComparisonCharts.innerHTML = '';
    }

    if (approvalRecommendationBox && approvalRecommendationTitle && approvalRecommendationBody) {
        approvalRecommendationBox.style.display = 'block';
        approvalRecommendationBox.className = 'recommendation-box cautious';
        approvalRecommendationTitle.textContent = '실행 옵션';
        approvalRecommendationBody.textContent = '현재 모델을 유지하거나, 재학습을 실행해 개선 여부를 다시 평가할 수 있습니다.';
    }

    if (approvalReportDetails) {
        approvalReportDetails.innerHTML = '';
        const details = report?.findings || report?.details || [];
        details.forEach((detail) => {
            const li = document.createElement('li');
            li.textContent = detail;
            approvalReportDetails.appendChild(li);
        });
    }
}

function renderApprovalProgress() {
    if (approvalOutcomeSeverityBadge) {
        approvalOutcomeSeverityBadge.textContent = '정리 중';
        approvalOutcomeSeverityBadge.className = 'toss-badge neutral';
    }
    if (approvalOutcomeTitle) {
        approvalOutcomeTitle.textContent = '재학습 전후 비교 보고서';
    }
    if (approvalOutcomeSubtitle) {
        approvalOutcomeSubtitle.textContent = '재학습을 실행하고 결과를 정리하고 있습니다.';
    }
    if (approvalOutcomeBeforeMape) approvalOutcomeBeforeMape.textContent = 'N/A';
    if (approvalOutcomeAfterMape) approvalOutcomeAfterMape.textContent = 'N/A';
    if (approvalOutcomeBeforeRmse) approvalOutcomeBeforeRmse.textContent = 'N/A';
    if (approvalOutcomeAfterRmse) approvalOutcomeAfterRmse.textContent = 'N/A';
    if (approvalOutcomeBanner) {
        approvalOutcomeBanner.style.background = '#f3f4f6';
        approvalOutcomeBanner.style.color = '#374151';
        approvalOutcomeBanner.textContent = '재학습을 실행하고 결과 보고서를 준비하고 있습니다. 잠시만 기다려주세요.';
    }
    hideLegacyApprovalOutcomeBlocks();
    if (approvalOutcomeCharts) {
        approvalOutcomeCharts.style.display = 'none';
        approvalOutcomeCharts.innerHTML = '';
    }
    if (approvalOutcomeRecommendationBox && approvalOutcomeRecommendationTitle && approvalOutcomeRecommendationBody) {
        approvalOutcomeRecommendationBox.style.display = 'block';
        approvalOutcomeRecommendationBox.className = 'recommendation-box';
        approvalOutcomeRecommendationTitle.textContent = '진행 상태';
        approvalOutcomeRecommendationBody.textContent = '현재 운영 모델과 재학습 결과를 정리하고 있습니다.';
    }
}

function renderApprovalOutcome(data) {
    const model = data?.model_1 || {};
    const preview = pendingComparisonPreview || {};
    const retrained = Boolean(model?.retrained);
    const redeployed = Boolean(model?.redeployed);
    const previousMape = Number.isFinite(model?.previous_mape) ? model.previous_mape : preview.active_holdout_metrics?.mape;
    const previousRmse = Number.isFinite(model?.previous_rmse) ? model.previous_rmse : preview.active_holdout_metrics?.rmse;
    const candidateMape = Number.isFinite(model?.candidate_mape) ? model.candidate_mape : preview.candidate_holdout_metrics?.mape;
    const candidateRmse = Number.isFinite(model?.candidate_rmse) ? model.candidate_rmse : preview.candidate_holdout_metrics?.rmse;
    const mapeImprovement = Number.isFinite(previousMape) && previousMape > 0 && Number.isFinite(candidateMape)
        ? ((previousMape - candidateMape) / previousMape) * 100
        : null;
    const rmseImprovement = Number.isFinite(previousRmse) && previousRmse > 0 && Number.isFinite(candidateRmse)
        ? ((previousRmse - candidateRmse) / previousRmse) * 100
        : null;
    const beforeLabel = '변경 전 모델';
    const afterLabel = redeployed ? '변경 후 모델' : '변경 후 후보 모델';
    const bannerColor = '#f3f4f6';
    const bannerTextColor = '#374151';
    const bannerTitle = redeployed
        ? '재학습 결과를 비교한 뒤 변경 후 모델을 운영 모델로 반영했습니다.'
        : '재학습 결과를 비교한 뒤 현재 운영 모델을 유지했습니다.';
    const improvementBits = [];
    if (Number.isFinite(mapeImprovement)) {
        improvementBits.push(`MAPE ${mapeImprovement >= 0 ? `${mapeImprovement.toFixed(2)}% 개선` : `${Math.abs(mapeImprovement).toFixed(2)}% 악화`}`);
    }
    if (Number.isFinite(rmseImprovement)) {
        improvementBits.push(`RMSE ${rmseImprovement >= 0 ? `${rmseImprovement.toFixed(2)}% 개선` : `${Math.abs(rmseImprovement).toFixed(2)}% 악화`}`);
    }
    const conclusion = redeployed
        ? `${beforeLabel} 대비 ${afterLabel}의 MAPE ${formatPercent(candidateMape)}, RMSE ${formatMetric(candidateRmse)}가 더 좋아 새 모델을 반영했습니다.`
        : `${beforeLabel} 대비 ${afterLabel}를 비교한 결과 개선이 충분하지 않아 현재 모델을 유지했습니다.`;
    const improvementText = improvementBits.join(' · ') || '향상 폭을 계산할 수 있는 비교 지표가 부족합니다.';

    if (approvalOutcomeSeverityBadge) {
        approvalOutcomeSeverityBadge.textContent = '비교 결과';
        approvalOutcomeSeverityBadge.className = 'toss-badge neutral';
    }
    if (approvalOutcomeTitle) {
        approvalOutcomeTitle.textContent = '재학습 전후 비교 보고서';
    }
    if (approvalOutcomeSubtitle) {
        approvalOutcomeSubtitle.textContent = conclusion;
    }
    if (approvalOutcomeBeforeMape) approvalOutcomeBeforeMape.textContent = formatPercent(previousMape);
    if (approvalOutcomeAfterMape) approvalOutcomeAfterMape.textContent = formatPercent(candidateMape);
    if (approvalOutcomeBeforeRmse) approvalOutcomeBeforeRmse.textContent = formatMetric(previousRmse);
    if (approvalOutcomeAfterRmse) approvalOutcomeAfterRmse.textContent = formatMetric(candidateRmse);

    if (approvalOutcomeBanner) {
        approvalOutcomeBanner.style.background = bannerColor;
        approvalOutcomeBanner.style.color = bannerTextColor;
        approvalOutcomeBanner.textContent = bannerTitle;
    }

    hideLegacyApprovalOutcomeBlocks();

    if (approvalOutcomeCharts) {
        const charts = [];
        if (preview.active_plot_b64) {
            charts.push(buildComparisonChartCard(
                '변경 전 모델 그래프',
                '재학습 실행 전 모델의 비교 구간 그래프입니다.',
                preview.active_plot_b64
            ));
        }
        if (preview.candidate_plot_b64) {
            charts.push(buildComparisonChartCard(
                redeployed ? '변경 후 모델 그래프' : '변경 후 후보 모델 그래프',
                redeployed ? '재학습 반영 후 모델의 비교 구간 그래프입니다.' : '재학습으로 만든 후보 모델의 비교 구간 그래프입니다.',
                preview.candidate_plot_b64
            ));
        }
        approvalOutcomeCharts.style.display = charts.length ? 'grid' : 'none';
        approvalOutcomeCharts.innerHTML = charts.join('');
    }

    if (approvalOutcomeRecommendationBox && approvalOutcomeRecommendationTitle && approvalOutcomeRecommendationBody) {
        approvalOutcomeRecommendationBox.style.display = 'block';
        approvalOutcomeRecommendationBox.className = 'recommendation-box neutral';
        approvalOutcomeRecommendationTitle.textContent = '한 줄 코멘트';
        approvalOutcomeRecommendationBody.textContent = improvementText;
    }
}

async function submitApproval(approved) {
    if (!pendingSavedFilename) {
        return;
    }

    if (!approved) {
        closeApprovalPopupIfOpen();
        pendingSavedFilename = null;
        pendingComparisonPreview = null;
        if (resultSummary) {
        resultSummary.textContent = '재학습을 보류하고 현재 운영 모델을 유지했습니다.';
    }
    hideRetrainingDecisionPanel();
    setActionStatus('재학습을 보류하고 현재 운영 모델을 유지합니다.', 'success');
    return;
}

    let shouldOpenResultModal = true;
    setButtonBusy(approveRetrainBtn, true, '재학습 중...');
    setButtonBusy(inlineRetrainBtn, true, '재학습 중...');
    if (rejectRetrainBtn) {
        rejectRetrainBtn.disabled = true;
        rejectRetrainBtn.classList.add('is-busy');
    }
    if (inlineKeepModelBtn) {
        inlineKeepModelBtn.disabled = true;
        inlineKeepModelBtn.classList.add('is-busy');
    }
    setActionStatus('재학습 후보를 탐색하고 있습니다. 잠시만 기다려주세요.', 'info');
    renderApprovalProgress();
    if (approvalResultModal) {
        setTimeout(() => {
            if (shouldOpenResultModal) {
                approvalResultModal.show();
            }
        }, 160);
    }

    try {
        const response = await fetch(`${API_ROOT}approve-retrain`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                saved_filename: pendingSavedFilename,
                approved: true,
            }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || '재학습 승인 처리 중 오류가 발생했습니다.');
        }

        if (approved) {
            closeApprovalPopupIfOpen();
            if (data.result_visualizing_LSTM && imageContainer) {
                const activeImg = imageContainer.querySelector('img[alt="현재 모델 예측 그래프"]');
                if (activeImg) {
                    activeImg.src = data.result_visualizing_LSTM;
                }
            }
            if (resultSummary) {
                resultSummary.textContent = data.model_1?.redeployed
                    ? '재학습 결과 후보 모델이 더 우수해 운영 모델이 갱신되었습니다.'
                    : '재학습은 수행됐지만 개선 폭이 충분하지 않아 현재 운영 모델을 유지했습니다.';
            }
            renderApprovalOutcome(data);
            hideRetrainingDecisionPanel();
            renderOverviewMetrics({
                targetColumn: data.model_1?.target_column,
                modelFamily: data.model_1?.model_family,
                thresholdRmse: data.model_1?.rmse_threshold,
                activeMape: data.model_1?.serving_metrics?.mape ?? data.model_1?.metrics?.forecast?.mape,
                activeRmse: data.model_1?.serving_metrics?.rmse ?? data.model_1?.metrics?.forecast?.rmse,
                baselineMape: null,
                baselineRmse: null,
                exceedPercent: null,
                approvalRequired: false,
                statusLabel: data.model_1?.redeployed ? '재학습 반영 완료' : '현재 모델 유지',
            });
            const servingMetrics = data.model_1?.serving_metrics || data.model_1?.metrics?.forecast || {};
            const currentThresholdRmse = data.model_1?.rmse_threshold;
            const currentRmse = servingMetrics?.rmse;
            const currentApprovalRequired = Number.isFinite(currentRmse) && Number.isFinite(currentThresholdRmse)
                ? currentRmse > currentThresholdRmse
                : false;
            renderCurrentModelReport({
                savedFilename: data.saved_filename,
                targetColumn: data.model_1?.target_column,
                modelName: data.model_1?.model_name || 'melbourne_deployed_model',
                thresholdRmse: currentThresholdRmse,
                mape: servingMetrics?.mape,
                rmse: servingMetrics?.rmse,
                mae: servingMetrics?.mae,
                statusLabel: data.model_1?.redeployed ? '재학습 반영 완료' : '현재 모델 유지',
                approvalRequired: currentApprovalRequired,
                timestamp: new Date().toLocaleString('ko-KR', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                }),
            });
            setActionStatus(
                data.model_1?.redeployed
                    ? '재학습이 완료되어 active 모델이 새 후보로 갱신되었습니다.'
                    : '재학습은 완료됐지만 기존 active 모델을 유지했습니다.',
                'success'
            );
        }
    } catch (error) {
        console.error(error);
        shouldOpenResultModal = false;
        if (approvalResultModal) {
            approvalResultModal.hide();
        }
        setActionStatus(error.message || '재학습 승인 처리 중 오류가 발생했습니다.', 'error');
    } finally {
        pendingSavedFilename = null;
        setButtonBusy(approveRetrainBtn, false, '재학습 중...');
        setButtonBusy(inlineRetrainBtn, false, '재학습 중...');
        if (rejectRetrainBtn) {
            rejectRetrainBtn.disabled = false;
            rejectRetrainBtn.classList.remove('is-busy');
        }
        if (inlineKeepModelBtn) {
            inlineKeepModelBtn.disabled = false;
            inlineKeepModelBtn.classList.remove('is-busy');
        }
    }
}

uploadForm?.addEventListener('submit', async function(event) {
    event.preventDefault();

    const file = fileUploadInput?.files?.[0];

    if (!file) {
        setActionStatus('업로드할 CSV 파일을 먼저 선택해주세요.', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setButtonBusy(uploadSubmitBtn, true, '업로드 중...');
    if (fileUploadInput) {
        fileUploadInput.disabled = true;
    }
    setActionStatus(`${file.name} 파일을 업로드하고 예측을 계산하고 있습니다. 잠시만 기다려주세요.`, 'info');

    try {
        const response = await fetch(`${API_ROOT}upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || '업로드 중 오류가 발생했습니다.');
        }

        renderUploadResult(data);

        if (data.approval_required) {
            pendingSavedFilename = data.saved_filename;
            pendingComparisonPreview = data.model_1?.comparison_preview || null;
            const popupOpened = openApprovalPopup(data.llm_report, pendingComparisonPreview);
            if (popupOpened) {
                setActionStatus('현재 모델 성능이 기준선을 초과했습니다. 재학습 검토 창을 새로 열었습니다.', 'info');
            } else {
                const activeModel = data.model_1 || {};
                const report = data.llm_report || activeModel.llm_report || {};
                const currentMape = metricValue(activeModel.metrics, 'mape');
                const thresholdRmse = Number.isFinite(activeModel.rmse_threshold)
                    ? activeModel.rmse_threshold
                    : report?.metrics?.threshold_rmse;
                const currentRmse = metricValue(activeModel.metrics, 'rmse');
                const exceedPercent = report?.metrics?.exceed_percent ?? (
                    Number.isFinite(currentRmse) && Number.isFinite(thresholdRmse) && thresholdRmse > 0
                        ? ((currentRmse / thresholdRmse) - 1) * 100
                        : null
                );
                renderApprovalReport(data.llm_report, pendingComparisonPreview);
                renderRetrainingDecisionPanel({
                    approvalRequired: true,
                    currentRmse,
                    currentMape,
                    thresholdRmse,
                    exceedPercent,
                });
                setActionStatus('팝업 창이 차단되어 현재 페이지에서 재학습 판단 패널을 표시합니다.', 'info');
            }
        } else {
            pendingComparisonPreview = null;
            closeApprovalPopupIfOpen();
            setActionStatus('업로드와 예측이 완료되었습니다.', 'success');
        }

        document.getElementById('performance')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
        console.error(error);
        setActionStatus(error.message || '업로드 중 오류가 발생했습니다.', 'error');
    } finally {
        setButtonBusy(uploadSubmitBtn, false, '업로드 중...');
        if (fileUploadInput) {
            fileUploadInput.disabled = false;
        }
    }
});

async function loadInitialDashboard() {
    try {
        const response = await fetch(`${API_ROOT}dashboard-summary`);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || '초기 대시보드 데이터를 불러오지 못했습니다.');
        }
        renderInitialDashboard(data);
    } catch (error) {
        console.error(error);
        if (initialStreamSummary) {
            initialStreamSummary.textContent = '초기 기준 시계열을 불러오지 못했습니다. 서버 상태를 확인한 뒤 새로고침해주세요.';
        }
    }
}

fileUploadInput?.addEventListener('change', () => {
    updateSelectedFileName(fileUploadInput?.files?.[0]?.name || '');
});

approveRetrainBtn?.addEventListener('click', () => submitApproval(true));
rejectRetrainBtn?.addEventListener('click', () => submitApproval(false));
inlineRetrainBtn?.addEventListener('click', () => submitApproval(true));
inlineKeepModelBtn?.addEventListener('click', () => submitApproval(false));
closeApprovalResultBtn?.addEventListener('click', () => {
    approvalResultModal?.hide();
    cleanupModalArtifacts();
});
approvalResultModalEl?.addEventListener('hidden.bs.modal', cleanupModalArtifacts);

loadInitialDashboard();
