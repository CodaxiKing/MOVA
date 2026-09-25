/* Connect the original canvas components to the local MOVA API. */
(function () {
  'use strict';
  var page = location.pathname.split('/').pop() || 'index.html';
  var runs = [], info = null;
  function api(path, options) {
    return fetch(path, options).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) throw new Error(body.error || 'HTTP ' + response.status);
        return body;
      });
    });
  }
  function fileData(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () { resolve({ name: file.name, data: String(reader.result).split(',')[1] }); };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }
  function loadRuns(instance) {
    return api('/api/runs').then(function (value) { runs = value; instance.setState({ liveLoaded: true }); })
      .catch(function (error) { instance.setState({ liveError: error.message }); });
  }
  function submit(instance, operation, reference, motion) {
    if (!motion || (operation === 'infer' && !reference)) {
      instance.setState({ liveError: 'Selecione os arquivos de entrada.' }); return;
    }
    instance.setState({ phase: 'running', liveError: null, liveStatus: 'Enviando arquivos…', progress: 0 });
    Promise.all([reference ? fileData(reference) : null, fileData(motion)]).then(function (files) {
      var body = { operation: operation, motion: files[1] };
      if (files[0]) body.reference = files[0];
      if (operation === 'infer') {
        body.model = instance.state.model;
        body.frames = instance.state.frames;
        body.steps = instance.state.steps;
        body.device = instance.state.device;
        body.precision = instance.state.precision;
        body.guidance = instance.state.guidance;
        body.conditioning = instance.state.cond;
        body.seed = instance.state.seed;
      }
      return api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    }).then(function (job) { poll(instance, job.id); }).catch(function (error) {
      instance.setState({ phase: 'idle', liveStatus: null, liveError: error.message });
    });
  }
  function poll(instance, id) {
    api('/api/jobs/' + id).then(function (job) {
      if (job.status === 'running') {
        instance.setState({ liveStatus: (job.log || []).slice(-1)[0] || 'Processando…', progress: 0 });
        setTimeout(function () { poll(instance, id); }, 1500);
      } else if (job.status === 'success') {
        instance.setState({ phase: 'done', liveStatus: 'Concluído · ' + job.run_id, liveOutput: job.output || null,
          liveRun: job.run_id, progress: 100 });
        loadRuns(instance);
        if (instance.state.history) instance.state.history.unshift({ mono: instance.state.model === 'wan' ? 'WAN' : 'TINY',
          bg: '#1c1c1f', fg: '#f4f4f5', meta: job.run_id });
      } else {
        instance.setState({ phase: 'idle', liveStatus: null, liveError: job.error || 'A execução falhou.' });
      }
    }).catch(function (error) { instance.setState({ phase: 'idle', liveError: error.message }); });
  }
  function setup(instance) {
    var original = instance.renderVals.bind(instance);
    instance.renderVals = function () {
      var v = original();
      if (!info) {
        v.hw = { name: 'Detectando máquina…', use: 'Lendo core.info', gpu: '—', gpuColor: '#a1a1aa', ram: '—', disk: '—', profile: '—', can: [] };
        v.machines = [{ label: 'Máquina atual', pressed: 'true', bg: '#d2ff3c', fg: '#0a0a0b', pick: function () {} }];
        v.authDisabled = true; v.authLabel = 'Verificando pesos'; return v;
      }
      var gpu = info.devices.find(function (d) { return d.type !== 'cpu'; });
      var cpu = info.devices.find(function (d) { return d.type === 'cpu'; }) || {};
      var memory = cpu.memory || {};
      var disk = info.disk_free_gb;
      v.machines = [{ label: 'Máquina atual', pressed: 'true', bg: '#d2ff3c', fg: '#0a0a0b', pick: function () {} }];
      v.hw = { name: gpu ? gpu.name : cpu.name || info.os, use: 'Ambiente detectado agora pelo MOVA.',
        gpu: gpu ? gpu.name + ' · ' + gpu.id : 'Sem GPU suportada detectada', gpuColor: gpu ? '#d2ff3c' : '#ff8cc4',
        ram: memory.free_gb + ' / ' + memory.total_gb + ' GB livres / total', disk: disk + ' GB livres',
        profile: gpu ? 'GPU · ' + (gpu.total_memory_gb || '?') + ' GB' : 'CPU',
        can: [['Extração MediaPipe', 'disponível'], ['Smoke test tiny', 'disponível'],
          ['Wan 1.3B', gpu ? 'verificar recursos' : 'sem GPU']].map(function (c) { return { label: c[0], state: c[1], dot: c[1] === 'sem GPU' ? '#ff4fa3' : '#d2ff3c' }; }) };
      v.disk = { label: disk + ' GB livres', pct: Math.min(100, Math.round(19.04 / Math.max(disk, 1) * 100)) + '%', color: disk >= 19.04 ? '#d2ff3c' : '#ff4fa3' };
      v.verdict = { title: 'Pesos: consulte o cache local', text: 'A interface não inicia downloads. O Estúdio usa apenas pesos já presentes e faz a checagem de recursos antes de inferir.',
        bg: '#1c1c1f', border: '#3a3a40', fg: '#f4f4f5', sub: '#a1a1aa' };
      v.authDisabled = true; v.authLabel = 'Download somente pela CLI'; v.authorize = function () {};
      return v;
    };
    instance.componentDidMount = function () {
      Promise.all([api('/api/info'),api('/api/runs')]).then(function (values) { info = values[0]; runs = values[1]; instance.setState({ loaded: true }); })
        .catch(function (error) { instance.setState({ liveError: error.message }); });
    };
  }
  function experiments(instance) {
    instance.renderVals = function () {
      var self = this, s = this.state;
      var filtered = runs.filter(function (r) { return s.filter === 'all' || r.kind === s.filter; });
      var current = runs.find(function (r) { return r.run_id === s.selected; }) || filtered[0] || null;
      function color(kind) { return kind === 'baseline' ? '#d2ff3c' : kind === 'train' ? '#ff4fa3' : '#f4f4f5'; }
      return {
        totalRuns:runs.length, extractRuns:runs.filter(function(r){return r.kind==='extract';}).length,
        tinyRuns:runs.filter(function(r){return r.kind==='baseline'&&String(r.model).indexOf('tiny')>=0;}).length,
        realRuns:runs.filter(function(r){return r.kind==='baseline'&&String(r.model).indexOf('tiny')<0&&r.status==='success';}).length,
        filters: [['all','Todos'],['extract','Extração'],['baseline','Baseline'],['train','Treino']].map(function (f) {
          var on = s.filter === f[0]; return { label:f[1], pressed:String(on), bg:on?'#d2ff3c':'transparent', fg:on?'#0a0a0b':'#f4f4f5',
            pick:function () { self.setState({ filter:f[0], selected:null }); } };
        }),
        rows: filtered.map(function (r) { var on = current && r.run_id === current.run_id, c = color(r.kind);
          return { id:r.run_id, kind:(r.kind||'run').toUpperCase(), kindBg:c, kindFg:c==='#f4f4f5'?'#0a0a0b':'#0a0a0b',
            model:r.model||'—', input:r.frames ? r.frames+' q' : '—', time:r.wall_time_s == null?'—':r.wall_time_s+' s',
            status:r.status||'—', statusColor:r.status==='success'?'#d2ff3c':r.status==='failed'?'#ff8cc4':'#a1a1aa',
            pressed:String(!!on), bg:on?'#1c1c1f':'transparent', mark:on?'#d2ff3c':'transparent',
            pick:function () { self.setState({selected:r.run_id}); } };
        }),
        sel: current ? { id:current.run_id, kind:(current.kind||'run').toUpperCase(), kindBg:color(current.kind), kindFg:'#0a0a0b',
          summary:current.error||('Status: '+current.status), link:current.media && current.media['output.mp4'] ? 'resultados.html?run='+encodeURIComponent(current.run_id) : current.record_url,
          cta:current.media && current.media['output.mp4'] ? 'Ver resultado' : 'Abrir run.json',
          fields:[['modelo',current.model],['status',current.status],['resolução',current.resolution],['quadros',current.frames],
            ['tempo',current.wall_time_s == null ? null : current.wall_time_s+' s'],['início',current.started_at]].map(function (f) { return {k:f[0],v:f[1]||'—'}; }),
          path:'experiments/runs/'+current.run_id+'/run.json' } : { id:'Nenhum run', kind:'—', kindBg:'#1c1c1f', kindFg:'#f4f4f5',
            summary:'Nenhum registro local disponível.', link:'#', cta:'Sem registro', fields:[], path:'experiments/runs/' }
      };
    };
    instance.componentDidMount = function () { loadRuns(instance); };
  }
  function studio(instance) {
    var original = instance.renderVals.bind(instance);
    instance.renderVals = function () {
      var v = original(), self = this, s = this.state;
      v.onRef = function (event) { var file = event.target.files[0]; if (file) { self.liveReference = file; self.setState({refUrl:URL.createObjectURL(file),refName:file.name,refSample:false,phase:'idle'}); } };
      v.onMotion = function (event) { var file = event.target.files[0]; if (file) { self.liveMotion = file; self.setState({motionUrl:URL.createObjectURL(file),motionName:file.name,motionSample:false,phase:'idle'}); } };
      v.useSamples = function () {
        Promise.all([fetch('/api/sample/reference'),fetch('/api/sample/motion')]).then(function (responses) {
          if (!responses[0].ok || !responses[1].ok) throw new Error('Exemplos locais indisponíveis.');
          return Promise.all(responses.map(function (r) { return r.blob(); }));
        }).then(function (blobs) {
          self.liveReference = new File([blobs[0]],'maya.png',{type:'image/png'});
          self.liveMotion = new File([blobs[1]],'dance.mp4',{type:'video/mp4'});
          self.setState({refUrl:URL.createObjectURL(blobs[0]),motionUrl:URL.createObjectURL(blobs[1]),
            refName:'maya.png',motionName:'dance.mp4',refSample:false,motionSample:false,phase:'idle'});
        }).catch(function (e) { self.setState({liveError:e.message}); });
      };
      v.generate = function () { submit(self,'infer',self.liveReference,self.liveMotion); };
      v.genDisabled = !self.liveReference || !self.liveMotion || s.phase === 'running';
      v.genLabel = s.phase === 'running' ? 'Gerando…' : 'Gerar';
      v.genHint = s.liveError || s.liveStatus || (s.model === 'tiny' ? 'tiny · saída de teste com ruído' : 'Wan · pesos locais · sem download');
      v.outFigure = false; v.outNoise = false; v.outIdle = !s.liveOutput && s.phase !== 'running';
      v.outCaption = s.liveOutput ? 'Saída real · '+s.liveRun : 'Saída · aguardando geração';
      v.busyStage = s.liveStatus || 'Processando…'; v.progressPct = s.phase === 'running' ? '…' : '0%';
      v.stages = v.stages.map(function (stage, i) { stage.status = s.phase === 'done' ? 'concluído' : s.phase === 'running' && i===0 ? 'em andamento' : 'aguardando'; return stage; });
      v.warnings = s.liveError ? [s.liveError] : [];
      v.history = s.history || []; v.noHistory = !v.history.length;
      v.command = 'mova infer --model '+s.model+' --reference <arquivo> --motion <arquivo> --device '+s.device+' --precision '+s.precision;
      v.liveOutput = s.liveOutput;
      return v;
    };
  }
  function motion(instance) {
    var original = instance.renderVals.bind(instance);
    instance.renderVals = function () {
      var v = original();
      v.sk = {lines:[],dots:[]}; v.face=[]; v.hands=[]; v.strip=[];
      return v;
    };
    instance.componentDidMount = function () { loadRuns(instance); };
  }
  function results(instance) {
    var original = instance.renderVals.bind(instance);
    instance.renderVals = function () { var v=original();v.ref={limbs:[],hands:[]};v.ctrl=[];v.out={limbs:[],hands:[]};
      var current=runs.find(function(r){return r.run_id===new URLSearchParams(location.search).get('run');});
      v.runLabel=current?current.run_id:'selecione um run';return v; };
    instance.componentDidMount = function () { loadRuns(instance); };
  }
  function ensure(parent, id, html) {
    if (!parent) return null;
    var node = parent.querySelector('#'+id);
    if (!node) { parent.insertAdjacentHTML('beforeend',html); node=parent.querySelector('#'+id); }
    return node;
  }
  function afterRender(instance, root) {
    if (page === 'experimentos.html') {
      var cards = root.querySelectorAll('[aria-labelledby="runs-h"] > div:nth-child(2) > div');
      var counts=[runs.length,runs.filter(function(r){return r.kind==='extract'}).length,
        runs.filter(function(r){return r.kind==='baseline'&&String(r.model).indexOf('tiny')>=0;}).length,
        runs.filter(function(r){return r.kind==='baseline'&&String(r.model).indexOf('tiny')<0&&r.status==='success'}).length];
      cards.forEach(function(card,i){var first=card.querySelector('span');if(first)first.textContent=counts[i];});
      root.querySelectorAll('[role="row"] [role="cell"]:last-child').forEach(function(cell,i){var r=runs.filter(function(x){return instance.state.filter==='all'||x.kind===instance.state.filter;})[i];if(r)cell.textContent=r.status;});
    }
    if (page === 'ambiente.html') {
      var models = root.querySelectorAll('article');
      if (info && models.length>=2) {
        var tiny=info.model_details.tiny, wan=info.model_details.wan;
        var t=models[0].querySelectorAll('dd'),w=models[1].querySelectorAll('dd');
        if(t[0])t[0].textContent=tiny.spec.verification['pytorch/cpu']||'UNVERIFIED';
        if(t[1])t[1].textContent=tiny.spec.verification['pytorch/cuda']||'UNVERIFIED';
        if(w[0])w[0].textContent=wan.spec.verification['pytorch/cuda']||'UNVERIFIED';
        var completed=runs.filter(function(r){return r.kind==='baseline'&&r.status==='success'&&String(r.model).indexOf('tiny')<0;});
        if(completed.length){
          var description=models[1].querySelector('span[style*="line-height: 1.5"]');
          if(description)description.textContent='Referência + controle de pose → vídeo. '+completed.length+' execução(ões) real(is) concluída(s) nesta máquina.';
          if(w[0])w[0].textContent='PASS em '+completed[0].run_id+' · ver Experimentos';
        }
      }
    }
    if (page === 'estudio.html') {
      var motionFigure=root.querySelector('main figure:nth-of-type(2)');
      if(motionFigure&&instance.state.motionUrl){var svg=motionFigure.querySelector('svg');if(svg)svg.outerHTML='<video class="mova-live-video" autoplay loop muted playsinline src="'+instance.state.motionUrl+'"></video>';}
      var extras=root.querySelectorAll('input[aria-label^="Imagem extra de identidade"]');
      extras.forEach(function(input){input.disabled=true;input.title='Identidade multi-imagem ainda não integrada ao core';var label=input.closest('label');if(label)label.style.opacity='.45';});
    }
    if (page === 'movimento.html') {
      var section=root.querySelector('[aria-labelledby="v-h"]');
      var control=ensure(section,'mova-motion-upload','<div id="mova-motion-upload" class="mova-live-upload"><input type="file" accept="video/*" aria-label="Vídeo para extração"><button type="button">Extrair</button><span class="mova-live-status"></span></div>');
      if(control){control.querySelector('button').onclick=function(){submit(instance,'preprocess',null,control.querySelector('input').files[0]);};
        control.querySelector('span').textContent=instance.state.liveError||instance.state.liveStatus||'Selecione um vídeo para extrair.';}
      var extracted=runs.filter(function(r){return r.kind==='extract'&&r.media&&r.media['pose_openpose.mp4'];});
      var current=extracted.find(function(r){return r.run_id===instance.state.liveRun;})||
        extracted.find(function(r){return r.summary&&r.summary.body&&r.summary.body.detection_rate>0;})||extracted[0];
      var preview=section&&section.querySelector('div[style*="height: 560px"]');
      var layer=instance.state.liveLayer||'pose_openpose.mp4';
      if(preview){preview.innerHTML=current&&current.media[layer]?'<video class="mova-live-video" controls src="'+current.media[layer]+'"></video>':'<div class="mova-live-empty">Aguardando vídeo de movimento</div>';}
      var group=section&&section.querySelector('[aria-label="Camadas"]');
      if(group){group.querySelectorAll('button').forEach(function(button,i){var names=['body_preview.mp4','face_preview.mp4','hands_preview.mp4'];button.disabled=!current||!current.media[names[i]];button.onclick=function(){instance.setState({liveLayer:names[i]});};});}
      var scrub=section&&section.querySelector('#mscrub');if(scrub)scrub.parentElement.style.display='none';
      var strip=root.querySelector('[aria-labelledby="strip-h"]');
      if(strip){var grid=strip.querySelector('div[style*="repeat(9"]');if(grid){grid.innerHTML=current?'<video class="mova-live-video" controls src="'+current.media['pose_openpose.mp4']+'" style="height:150px;width:100%;grid-column:1/-1"></video>':'<div class="mova-live-empty" style="grid-column:1/-1">Nenhum controle registrado</div>';}}
      var cards=root.querySelectorAll('article');if(current&&cards.length>=3){
        var summary=current.summary||{};var metrics=[summary.body&&summary.body.detection_rate,summary.face&&summary.face.detection_rate,summary.hands&&summary.hands.detection_rate_left];
        cards.forEach(function(card,i){var old=card.querySelector('.mova-live-status');if(!old){old=document.createElement('span');old.className='mova-live-status';card.appendChild(old);}old.textContent='Detecção no run '+current.run_id+': '+(metrics[i]==null?'—':Math.round(metrics[i]*100)+'%');});
      }
    }
    if (page === 'resultados.html') {
      var available=runs.filter(function(r){return r.media&&r.media['output.mp4'];});
      var requested=new URLSearchParams(location.search).get('run');
      var selected=available.find(function(r){return r.run_id===requested;})||available[0];
      var panel=root.querySelector('[aria-labelledby="cmp-h"]');
      if(panel){
        var picker=ensure(panel,'mova-result-picker','<select id="mova-result-picker" aria-label="Escolher resultado" style="background:#1c1c1f;color:#f4f4f5;border:1px solid #3a3a40;border-radius:12px;padding:8px"></select>');
        if(picker){picker.innerHTML=available.map(function(r){return '<option value="'+r.run_id+'">'+r.run_id+'</option>';}).join('');if(selected)picker.value=selected.run_id;
          picker.onchange=function(){location.href='resultados.html?run='+encodeURIComponent(picker.value);};}
        var figures=panel.querySelectorAll('figure > div');
        if(figures.length===3){
          figures[0].innerHTML=selected&&selected.input_media&&selected.input_media.reference?'<img class="mova-live-video" src="'+selected.input_media.reference+'" alt="Referência usada na execução">':'<div class="mova-live-empty">Referência indisponível</div>';
          figures[1].innerHTML=selected&&selected.media['control.mp4']?'<video class="mova-live-video" controls src="'+selected.media['control.mp4']+'"></video>':'<div class="mova-live-empty">Controle indisponível</div>';
          figures[2].innerHTML=selected?'<video class="mova-live-video" controls src="'+selected.media['output.mp4']+'"></video>':'<div class="mova-live-empty">Nenhum vídeo gerado</div>';
        }
        var label=panel.querySelector('div:first-child > span');if(label)label.textContent=selected?selected.run_id:'Sem resultado';
        var playback=panel.querySelector('#rscrub');if(playback)playback.parentElement.style.display='none';
      }
      var int=root.querySelector('[aria-labelledby="int-h"]');if(int){var dd=int.querySelectorAll('dd');dd.forEach(function(node){node.textContent='—';});
        var validation=selected&&selected.stats&&selected.stats.output_validation;
        if(validation&&dd.length>=8){dd[1].textContent=validation.num_frames;dd[3].textContent=validation.fps;dd[5].textContent=validation.width+'×'+validation.height;dd[7].textContent=validation.status;}}
      var met=root.querySelector('[aria-labelledby="met-h"]');if(met){var values=met.querySelectorAll('dd');values.forEach(function(node){node.textContent='pendente';});}
      var review=root.querySelector('[aria-labelledby="rev-h"]');if(review){var buttons=review.querySelectorAll('button');var notes=review.querySelector('textarea');
        if(notes&&selected&&document.activeElement!==notes)notes.value=(selected.web_review||{}).notes||'';
        function save(verdict){if(!selected)return;var value={verdict:verdict,notes:notes.value};selected.web_review=value;
          api('/api/runs/'+selected.run_id+'/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}).catch(function(e){notes.title=e.message;});}
        buttons.forEach(function(b,i){b.disabled=!selected;b.title=selected?'Salvar revisão no registro':'Sem resultado';b.onclick=function(){save(i===0?'ok':'bad');};});
        if(notes){notes.onchange=function(){save((selected.web_review||{}).verdict||null);};}
      }
    }
    if (page === 'index.html') {
      var baseline=root.querySelector('[aria-labelledby="st-h"] > div:nth-child(2) > div:nth-child(3) span:last-child');
      if(baseline)baseline.textContent='pipeline operacional; consulte Experimentos';
      var status=root.querySelector('[aria-labelledby="st-h"] > div:nth-child(2) > div:nth-child(4) span:last-child');
      if(status)status.textContent='protótipo em CPU';
    }
  }
  window.MOVALive={
    attach:function(instance){
      if(page==='ambiente.html')setup(instance);
      else if(page==='experimentos.html')experiments(instance);
      else if(page==='estudio.html')studio(instance);
      else if(page==='movimento.html')motion(instance);
      else if(page==='resultados.html')results(instance);
    },
    afterRender:afterRender
  };
})();
