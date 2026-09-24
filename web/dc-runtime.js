/*
 * Runtime mínimo para as telas exportadas do canvas de design (design/canvas/project/*.dc.html).
 * Substitui o support.js do editor: interpola {{expr}}, expande <sc-for>/<sc-if>, liga onClick/onChange
 * e roda a classe Component (DCLogic) de cada tela. Re-renderiza com um morph de DOM simples para
 * preservar foco e arrasto de sliders.
 */
(function () {
  'use strict';

  class DCLogic {
    constructor(props) { this.props = props || {}; this.state = {}; }
    setState(update) {
      Object.assign(this.state, typeof update === 'function' ? update(this.state) : update);
      schedule();
    }
    renderVals() { return {}; }
  }

  var EXPR = /\{\{([\s\S]+?)\}\}/g;
  var ONLY_EXPR = /^\s*\{\{([\s\S]+?)\}\}\s*$/;
  var fnCache = {};

  function evaluate(expr, scope) {
    var fn = fnCache[expr] || (fnCache[expr] = new Function('scope', 'with (scope) { return (' + expr + '); }'));
    try { return fn(scope); } catch (e) { return undefined; }
  }

  function interpolate(text, scope) {
    return text.replace(EXPR, function (_, expr) {
      var v = evaluate(expr, scope);
      return v == null ? '' : String(v);
    });
  }

  var BOOL_ATTRS = { disabled: 1, checked: 1, selected: 1, hidden: 1, readonly: 1, required: 1 };

  // Renderiza nós do template para uma lista de nós reais.
  function renderNodes(nodes, scope, out) {
    for (var i = 0; i < nodes.length; i++) renderNode(nodes[i], scope, out);
    return out;
  }

  function renderNode(t, scope, out) {
    if (t.nodeType === 3) {
      out.push(document.createTextNode(t.data.indexOf('{{') < 0 ? t.data : interpolate(t.data, scope)));
      return;
    }
    if (t.nodeType !== 1) return;
    var tag = t.localName;
    if (tag === 'sc-for') {
      var list = evaluate(t.getAttribute('list').replace(ONLY_EXPR, '$1'), scope) || [];
      var as = t.getAttribute('as') || 'item';
      for (var i = 0; i < list.length; i++) {
        var child = Object.create(scope);
        child[as] = list[i];
        renderNodes(t.childNodes, child, out);
      }
      return;
    }
    if (tag === 'sc-if') {
      if (evaluate(t.getAttribute('value').replace(ONLY_EXPR, '$1'), scope)) renderNodes(t.childNodes, scope, out);
      return;
    }
    var el = t.namespaceURI === 'http://www.w3.org/1999/xhtml'
      ? document.createElement(tag) : document.createElementNS(t.namespaceURI, t.tagName);
    el.__on = {};
    for (var a = 0; a < t.attributes.length; a++) {
      var attr = t.attributes[a], name = attr.name, val = attr.value;
      if (name.indexOf('hint-') === 0) continue;
      var lower = name.toLowerCase();
      if (lower === 'onclick' || lower === 'onchange') {
        var m = ONLY_EXPR.exec(val);
        if (m) el.__on[lower.slice(2)] = evaluate(m[1], scope);
        continue;
      }
      var only = ONLY_EXPR.exec(val);
      if (only) {
        var v = evaluate(only[1], scope);
        if (BOOL_ATTRS[lower]) { if (v) el.setAttribute(name, ''); continue; }
        if (v == null || v === false) continue;
        el.setAttribute(name, String(v));
      } else {
        el.setAttribute(name, val.indexOf('{{') < 0 ? val : interpolate(val, scope));
      }
    }
    var kids = renderNodes((t.content || t).childNodes, scope, []);
    for (var k = 0; k < kids.length; k++) el.appendChild(kids[k]);
    out.push(el);
  }

  // Eventos: um listener por elemento, que chama o handler atual (trocado a cada render).
  function bind(el) {
    if (el.__bound) return;
    el.__bound = true;
    el.addEventListener('click', function (e) { var h = el.__on && el.__on.click; if (h) h(e); });
    var changeEvt = el.localName === 'input' && el.type !== 'file' && el.type !== 'checkbox' && el.type !== 'radio' ? 'input' : 'change';
    el.addEventListener(changeEvt, function (e) { var h = el.__on && el.__on.change; if (h) h(e); });
  }

  function hasHandlers(el) { return !!(el.__on && (el.__on.click || el.__on.change)); }

  function sameKind(a, b) {
    return a.nodeType === b.nodeType && (a.nodeType !== 1 || (a.namespaceURI === b.namespaceURI && a.localName === b.localName));
  }

  function syncProps(cur) {
    if (cur.localName === 'input' && cur.type !== 'file') {
      var v = cur.getAttribute('value');
      if (v != null && cur.value !== v && document.activeElement !== cur) cur.value = v;
      else if (v != null && cur.value !== v && cur.type === 'range') cur.value = v;
    }
    if (cur.localName === 'video' && cur.hasAttribute('autoplay') && cur.paused) { var p = cur.play(); if (p && p.catch) p.catch(function () {}); }
  }

  function morphChildren(parent, next) {
    var cur = parent.childNodes;
    for (var i = 0; i < next.length; i++) {
      var c = cur[i], n = next[i];
      if (!c) { parent.appendChild(n); finalize(n); continue; }
      if (!sameKind(c, n) || (c.nodeType === 1 && (c.localName === 'img' || c.localName === 'video') && c.getAttribute('src') !== n.getAttribute('src'))) {
        parent.replaceChild(n, c); finalize(n); continue;
      }
      if (c.nodeType === 3) { if (c.data !== n.data) c.data = n.data; continue; }
      morphAttrs(c, n);
      c.__on = n.__on;
      if (hasHandlers(c)) bind(c);
      morphChildren(c, Array.prototype.slice.call(n.childNodes));
      syncProps(c);
    }
    while (cur.length > next.length) parent.removeChild(parent.lastChild);
  }

  function morphAttrs(c, n) {
    var i, a;
    for (i = c.attributes.length - 1; i >= 0; i--) { a = c.attributes[i]; if (!n.hasAttribute(a.name)) c.removeAttribute(a.name); }
    for (i = 0; i < n.attributes.length; i++) { a = n.attributes[i]; if (c.getAttribute(a.name) !== a.value) c.setAttribute(a.name, a.value); }
    if ('disabled' in c) c.disabled = n.hasAttribute('disabled');
  }

  function finalize(node) {
    if (node.nodeType !== 1) return;
    if (hasHandlers(node)) bind(node);
    syncProps(node);
    for (var i = 0; i < node.children.length; i++) finalize(node.children[i]);
  }

  var instance, templateNodes, root, pending = false;

  function render() {
    pending = false;
    var vals = instance.renderVals() || {};
    var scope = Object.assign(Object.create(null), vals);
    morphChildren(root, renderNodes(templateNodes, scope, []));
  }

  function schedule() {
    if (pending || !instance) return;
    pending = true;
    requestAnimationFrame(render);
  }

  // Ajusta a largura fixa das telas (1440 px) à janela, como no canvas.
  function fit() {
    var page = root.firstElementChild;
    if (!page) return;
    var w = parseFloat(page.style.width) || 1440;
    var z = Math.min(1, document.documentElement.clientWidth / w);
    root.style.zoom = z < 1 ? String(z) : '';
  }

  function boot() {
    var tpl = document.getElementById('dc-template');
    var src = document.querySelector('script[type="text/x-dc"]');
    root = document.getElementById('dc-root');
    templateNodes = Array.prototype.slice.call(tpl.content.childNodes);
    var props = {};
    try { props = JSON.parse(src.getAttribute('data-props') || '{}'); } catch (e) {}
    var Component = new Function('DCLogic', src.textContent + '\nreturn Component;')(DCLogic);
    instance = new Component(props);
    render();
    fit();
    window.addEventListener('resize', fit);
    if (instance.componentDidMount) instance.componentDidMount();
    window.addEventListener('pagehide', function () { if (instance.componentWillUnmount) instance.componentWillUnmount(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
