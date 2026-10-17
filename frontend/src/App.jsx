import React, { useEffect, useMemo, useState } from 'react'
import { api } from './api.js'
import Viewer from './Viewer.jsx'

const FLOW = [
  {id:'home', n:1, label:'首页'},
  {id:'case', n:2, label:'订单 / 定位'},
  {id:'design', n:3, label:'自动设计 / 导出'},
]

function Topbar({ step, caseData, busy, goHome }) {
  return <header className="topbar">
    <button className="brand" onClick={goHome}><span className="logo">Y</span><b>AUTO_YangGuan</b></button>
    <div className="crumb">{caseData ? `${caseData.name} / FDI ${caseData.fdi}` : '请选择订单'}</div>
    <span className={`state ${busy ? 'busy' : ''}`}>{busy ? '处理中…' : '就绪'}</span>
  </header>
}

function Stepper({ step }) {
  return <div className="stepper">{FLOW.map(item => <div key={item.id} className={`step ${step === item.id ? 'active' : ''} ${FLOW.findIndex(x=>x.id===step) > FLOW.findIndex(x=>x.id===item.id) ? 'done' : ''}`}>
    <span>{item.n}</span><b>{item.label}</b>
  </div>)}</div>
}

function Home({ testCases, onPickTest, onUpload, busy }) {
  const [fdi, setFdi] = useState('36')
  const [upper, setUpper] = useState(null)
  const [lower, setLower] = useState(null)
  return <div className="page intake">
    <section className="hero">
      <div><small>AUTO CROWN MVP</small><h1>牙冠自动生成测试台</h1><p>先完成基台定位、Margin、插入轴和 PatientFrame 复核，再允许生成外冠。</p></div>
      <div className="hero-badge">定位优先 · 保守生成</div>
    </section>

    <div className="grid two">
      <section className="panel">
        <div className="panel-head"><div><small>TEST DATA</small><h2>仓库测试病例</h2></div><span>{testCases.length} 例</span></div>
        <div className="case-list">
          {testCases.map(item => <button disabled={busy} key={item.name} className="case-card" onClick={() => onPickTest(item)}>
            <span className="fdi">{item.fdi}</span><div><b>{item.name}</b><small>{item.files.join(' · ')}</small></div><em>进入订单 →</em>
          </button>)}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head"><div><small>NEW ORDER</small><h2>上传本地病例</h2></div></div>
        <label className="field"><span>目标牙位 FDI</span><input value={fdi} onChange={e=>setFdi(e.target.value.replace(/\D/g,'').slice(0,2))} /></label>
        <label className="upload"><span>上颌 STL</span><input type="file" accept=".stl,.STL" onChange={e=>setUpper(e.target.files?.[0]||null)} /><b>{upper?.name || '选择文件'}</b></label>
        <label className="upload"><span>下颌 STL</span><input type="file" accept=".stl,.STL" onChange={e=>setLower(e.target.files?.[0]||null)} /><b>{lower?.name || '选择文件'}</b></label>
        <button className="primary full" disabled={busy || !upper || !lower || fdi.length !== 2} onClick={() => onUpload({fdi, upper, lower})}>创建订单</button>
      </section>
    </div>
  </div>
}

function LocalizationSummary({ localization }) {
  if (!localization) return <div className="notice">尚未定位。先执行“1. 基台定位 / Margin / 插入轴”。</div>
  const a = localization.abutment || {}
  const axis = localization.axis || {}
  const margin = localization.margin || {}
  const frame = localization.frame || {}
  return <>
    <div className="strategy">
      <div><b>基台</b><span>{a.status || '—'} · confidence {a.confidence || '—'}</span></div>
      <div><b>Margin</b><span>{margin.point_count || localization.margin_points?.length || 0} 点 · 自动工程候选</span></div>
      <div><b>插入轴 Z</b><span>{axis.status || '—'} · confidence {axis.confidence != null ? Number(axis.confidence).toFixed(2) : '—'}</span></div>
      <div><b>PatientFrame</b><span>det {frame.handedness != null ? Number(frame.handedness).toFixed(3) : '—'} · X近远中/Y颊侧/Z插入</span></div>
    </div>
    {axis.reasons?.length > 0 && <div className="notice">轴复核提示：{axis.reasons.join(' / ')}</div>}
  </>
}

function CasePage({ caseData, onLocalize, onGenerate, busy, message }) {
  const [visible, setVisible] = useState({target:true, opponent:true, support:true, margin:true, frame:true})
  return <div className="page order-page">
    <div className="page-title"><div><small>NEW CROWN ORDER</small><h1>新建牙冠病例</h1><p>必须先确认基台位置、Margin 与插入轴方向正确，再进入模板牙摆放。</p></div><div className="order-meta"><span>FDI</span><b>{caseData.fdi}</b></div></div>
    <div className="grid order-grid">
      <section className="panel preview-panel">
        <div className="panel-head"><div><small>GEOMETRY REVIEW</small><h2>定位与方向复核</h2></div></div>
        <Viewer assets={caseData.assets} localization={caseData.localization} visible={visible} />
        <div className="strategy">
          <div><b style={{color:'#39d98a'}}>Margin</b><span>绿色闭环</span></div>
          <div><b style={{color:'#ff6b6b'}}>X</b><span>近远中轴</span></div>
          <div><b style={{color:'#4dabf7'}}>Y</b><span>颊侧轴</span></div>
          <div><b style={{color:'#e599f7'}}>Z</b><span>插入轴 / 咬合方向</span></div>
        </div>
        {caseData.localized && <div className="metrics">
          {[
            ['support','基台'],['margin','Margin'],['frame','XYZ坐标']
          ].map(([key,label]) => <label className="toggle" key={key}><input type="checkbox" checked={visible[key] !== false} onChange={e=>setVisible(v=>({...v,[key]:e.target.checked}))}/><span>{label}</span></label>)}
        </div>}
      </section>
      <section className="panel controls-panel">
        <div className="panel-head"><div><small>LOCALIZATION FIRST</small><h2>生成前置条件</h2></div></div>
        <div className="template-card"><div className="tooth-icon">{caseData.fdi}</div><div><b>FDI_{caseData.fdi}.stl</b><small>只有定位通过后才会转成 canonical crown prior</small></div><span>已选择</span></div>
        <LocalizationSummary localization={caseData.localization} />
        <button className="secondary full" disabled={busy} onClick={onLocalize}>{busy ? '处理中…' : caseData.localized ? '重新定位并复核' : '1. 基台定位 / Margin / 插入轴'}</button>
        <div className="notice">{message || (caseData.localized ? '请在左侧确认棕色基台、绿色 Margin 和紫色 Z 轴方向正确。' : '先执行定位。')}</div>
        <button className="primary generate" disabled={busy || !caseData.localized} onClick={onGenerate}>{busy ? '正在生成牙冠…' : '2. 牙冠生成'}</button>
        {!caseData.localized && <small>未完成定位时禁止生成，避免再次出现模板牙方向错误。</small>}
      </section>
    </div>
  </div>
}

function DesignPage({ caseData, onBack }) {
  const [visible, setVisible] = useState({target:true, opponent:true, support:true, mesial:false, distal:false, margin:true, frame:true, outer:true})
  const report = caseData.report || {}
  const qc = report.outer_qc || {}
  const loc = caseData.localization || {}
  return <div className="page design-page">
    <div className="design-toolbar">
      <button className="secondary" onClick={onBack}>← 返回订单 / 定位</button>
      <div><b>{caseData.name}</b><small>FDI {caseData.fdi} · {caseData.status}</small></div>
      {caseData.assets.outer && <a className="primary export" href={caseData.assets.outer} download>导出 Outer STL</a>}
    </div>
    <div className="design-layout">
      <section className="viewer-panel"><Viewer assets={caseData.assets} localization={caseData.localization} visible={visible} /></section>
      <aside className="side-panel">
        <div><small>DISPLAY</small><h3>显示</h3></div>
        {[
          ['target','目标颌'],['opponent','对颌'],['support','基台'],['mesial','近中参考'],['distal','远中参考'],['margin','Margin'],['frame','XYZ / 插入轴'],['outer','自动外冠']
        ].map(([key,label]) => <label className="toggle" key={key}><input type="checkbox" checked={visible[key] !== false} onChange={e=>setVisible(v=>({...v,[key]:e.target.checked}))}/><span>{label}</span></label>)}
        <hr/>
        <div><small>LOCALIZATION</small><h3>定位</h3></div>
        <div className="metrics">
          <div><span>基台</span><b>{loc.abutment?.confidence || '—'}</b></div>
          <div><span>插入轴</span><b>{loc.axis?.status || '—'}</b></div>
          <div><span>轴置信度</span><b>{loc.axis?.confidence != null ? Number(loc.axis.confidence).toFixed(2) : '—'}</b></div>
          <div><span>Margin 点</span><b>{loc.margin_points?.length || '—'}</b></div>
        </div>
        <hr/>
        <div><small>QC</small><h3>工程检查</h3></div>
        <div className="metrics">
          <div><span>状态</span><b>{qc.status || caseData.status}</b></div>
          <div><span>最大位移</span><b>{qc.max_displacement_mm != null ? `${Number(qc.max_displacement_mm).toFixed(2)} mm` : '—'}</b></div>
          <div><span>P95 位移</span><b>{qc.p95_displacement_mm != null ? `${Number(qc.p95_displacement_mm).toFixed(2)} mm` : '—'}</b></div>
        </div>
        <div className="warn"><b>当前测试边界</b><p>基台定位、自动 Margin、插入轴、PatientFrame、邻牙参考与保守外冠已经接入。自动 Margin/Axis 仍是工程候选，必须视觉复核；正式内冠尚未纳入本 MVP。</p></div>
        {caseData.assets.localization && <a className="link" href={caseData.assets.localization} target="_blank">打开 localization.json</a>}
        {caseData.assets.report && <a className="link" href={caseData.assets.report} target="_blank">打开 report.json</a>}
      </aside>
    </div>
  </div>
}

export default function App() {
  const [step, setStep] = useState('home')
  const [testCases, setTestCases] = useState([])
  const [caseData, setCaseData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => { api.testCases().then(x => setTestCases(x.cases || [])).catch(e => setMessage(e.message)) }, [])
  const title = useMemo(() => FLOW.find(x=>x.id===step)?.label || '', [step])

  async function pickTest(item) {
    setBusy(true); setMessage(`正在载入 ${item.name}…`)
    try { const c = await api.createFromTestdata(item.name); setCaseData(c); setStep('case'); setMessage('病例已载入。先执行基台定位。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  async function upload(payload) {
    setBusy(true); setMessage('正在上传病例…')
    try { const c = await api.uploadCase(payload); setCaseData(c); setStep('case'); setMessage('病例已载入。先执行基台定位。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  async function localize() {
    if (!caseData) return
    setBusy(true); setMessage('正在执行：基台候选排名 → 粗轴 → Margin → 最终插入轴 → PatientFrame…')
    try { const c = await api.localize(caseData.id); setCaseData(c); setMessage(c.localization?.status === 'READY' ? '定位完成，请检查基台、Margin 与 XYZ 方向。' : '定位完成但存在复核项，请重点检查基台和紫色 Z 轴。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  async function generate() {
    if (!caseData?.localized) return
    setBusy(true); setMessage('正在执行：canonical 模板 → Margin 放置 → 近远中小范围适配 → 对颌避让 → QC…')
    try { const c = await api.generate(caseData.id); setCaseData(c); setStep('design'); setMessage('外冠候选已生成。') }
    catch(e){ setMessage(e.message) } finally { setBusy(false) }
  }
  function home(){ setStep('home'); setCaseData(null); setMessage('') }

  return <div className="app">
    <Topbar step={step} caseData={caseData} busy={busy} goHome={home}/>
    {step !== 'home' && <Stepper step={step}/>} 
    <main>
      {step === 'home' && <Home testCases={testCases} onPickTest={pickTest} onUpload={upload} busy={busy}/>} 
      {step === 'case' && caseData && <CasePage caseData={caseData} onLocalize={localize} onGenerate={generate} busy={busy} message={message}/>} 
      {step === 'design' && caseData && <DesignPage caseData={caseData} onBack={()=>setStep('case')}/>} 
    </main>
    {message && <div className="toast">{message}<button onClick={()=>setMessage('')}>×</button></div>}
    {busy && <div className="overlay"><div className="loader"></div><b>正在处理病例</b><span>{message}</span></div>}
    <footer className="footer"><span>AUTO_YangGuan · {title}</span><span>本地研发测试，不用于临床</span></footer>
  </div>
}
