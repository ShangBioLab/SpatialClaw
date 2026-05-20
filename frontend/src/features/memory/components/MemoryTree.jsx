import React, { useState, useEffect, useRef } from 'react';
import { ChevronRight, FileText, Database } from 'lucide-react';
import clsx from 'clsx';
import { api } from '../../../lib/memoryApi';
import { formatLeafLabel } from '../../../shared/utils/content';

const TreeNode = ({ domain, path, name, childrenCount, activeDomain, activePath, onNavigate, level, maxDepth }) => {
  const isAncestor = activeDomain === domain && activePath.startsWith(path + '/');
  const isActive = activeDomain === domain && activePath === path;
  
  const [expanded, setExpanded] = useState(isAncestor || isActive);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetched, setFetched] = useState(false);

  const prevActivePath = useRef(activePath);
  const prevActiveDomain = useRef(activeDomain);

  const canExpand = maxDepth === undefined || level < maxDepth;
  const hasChildren = canExpand && (fetched ? children.length > 0 : (childrenCount === undefined || childrenCount > 0));

  useEffect(() => {
    if (expanded && !fetched && hasChildren) {
      fetchChildren();
    }
  }, [expanded, fetched, hasChildren]);

  useEffect(() => {
    const pathChanged = activePath !== prevActivePath.current || activeDomain !== prevActiveDomain.current;
    if (pathChanged && (isAncestor || isActive) && !expanded) {
      setExpanded(true);
    }
    prevActivePath.current = activePath;
    prevActiveDomain.current = activeDomain;
  }, [activePath, activeDomain, isAncestor, isActive, expanded]);

  const fetchChildren = async () => {
    setLoading(true);
    try {
      const res = await api.get('/browse/node', { params: { domain, path, nav_only: true } });
      setChildren(res.data.children);
      setFetched(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClick = (e) => {
    e.stopPropagation();
    if (isActive) {
      if (hasChildren) setExpanded(!expanded);
    } else {
      onNavigate(path, domain);
      if (!expanded && hasChildren) setExpanded(true);
    }
  };

  return (
    <div>
      <div 
        className={clsx(
          "flex items-center gap-1.5 py-2 pr-2 rounded-xl text-sm transition-all duration-300 cursor-pointer group border border-transparent",
          isActive ? "bg-indigo-50 border-indigo-200 text-indigo-700 shadow-sm" : "text-slate-500 hover:bg-slate-50 hover:border-slate-200 hover:text-slate-800"
        )}
        style={{ paddingLeft: `${level * 12 + 8}px` }}
        onClick={handleClick}
      >
        <div 
          className="w-5 h-5 flex items-center justify-center flex-shrink-0"
          onClick={(e) => {
             if (hasChildren) {
                 e.stopPropagation();
                 setExpanded(!expanded);
             }
          }}
        >
          {loading ? (
            <div className="w-3 h-3 border-2 border-slate-500 border-t-transparent rounded-full animate-spin" />
          ) : hasChildren ? (
            <ChevronRight size={14} className={clsx("transition-transform text-slate-500 group-hover:text-slate-700", expanded && "rotate-90")} />
          ) : null}
        </div>
        <FileText size={14} className={clsx("flex-shrink-0", isActive ? "text-indigo-600" : "text-slate-400 group-hover:text-slate-600")} />
        <span className="truncate flex-1 text-[13px]">{formatLeafLabel(path, name)}</span>
      </div>
      
      {expanded && children.length > 0 && (
        <div>
          {children.map(child => (
            <TreeNode 
              key={child.path}
              domain={domain}
              path={child.path}
              name={child.name}
              childrenCount={child.approx_children_count}
              activeDomain={activeDomain}
              activePath={activePath}
              onNavigate={onNavigate}
              level={level + 1}
              maxDepth={maxDepth}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const MemoryTree = ({ domain, displayName, rootCount, activeDomain, activePath, onNavigate, maxDepth }) => {
  const [expanded, setExpanded] = useState(activeDomain === domain);
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fetched, setFetched] = useState(false);

  const prevActiveDomain = useRef(activeDomain);
  const prevActivePath = useRef(activePath);

  const hasChildren = fetched ? children.length > 0 : (rootCount === undefined || rootCount > 0);

  useEffect(() => {
    if (expanded && !fetched && hasChildren) {
      fetchChildren();
    }
  }, [expanded, fetched, hasChildren]);

  useEffect(() => {
    const changed = activeDomain !== prevActiveDomain.current || activePath !== prevActivePath.current;
    if (changed && activeDomain === domain && !expanded) {
      setExpanded(true);
    }
    prevActiveDomain.current = activeDomain;
    prevActivePath.current = activePath;
  }, [activeDomain, activePath, domain, expanded]);

  const fetchChildren = async () => {
    setLoading(true);
    try {
      const res = await api.get('/browse/node', { params: { domain, path: '', nav_only: true } });
      setChildren(res.data.children);
      setFetched(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const isActive = activeDomain === domain && activePath === '';

  const handleClick = (e) => {
    e.stopPropagation();
    if (isActive) {
      if (hasChildren) setExpanded(!expanded);
    } else {
      onNavigate('', domain);
      if (!expanded && hasChildren) setExpanded(true);
    }
  };

  return (
    <div className="mb-2">
      <div 
        className={clsx(
          "glass-card flex items-center gap-2 px-3 py-2.5 rounded-xl text-sm cursor-pointer group border border-transparent hover:!translate-y-0 relative overflow-hidden",
          isActive ? "bg-indigo-50 border-indigo-200 text-indigo-700 shadow-sm" : "bg-white text-slate-500 hover:bg-slate-50 hover:border-slate-200 hover:text-slate-800"
        )}
        onClick={handleClick}
      >
        <div 
          className="w-5 h-5 flex items-center justify-center flex-shrink-0"
          onClick={(e) => {
             if (hasChildren) {
                 e.stopPropagation();
                 setExpanded(!expanded);
             }
          }}
        >
          {loading ? (
            <div className="w-3.5 h-3.5 border-2 border-slate-500 border-t-transparent rounded-full animate-spin" />
          ) : hasChildren ? (
            <ChevronRight size={16} className={clsx("transition-transform text-slate-500 group-hover:text-slate-700", expanded && "rotate-90")} />
          ) : null}
        </div>
        <Database size={16} className={clsx("flex-shrink-0 ml-0.5", isActive ? "text-indigo-600" : "text-slate-500")} />
        <span className="font-medium flex-1 truncate ml-1">
          {displayName || `${domain.charAt(0).toUpperCase() + domain.slice(1)} Memory`}
        </span>
        {rootCount !== undefined && (
          <span className="text-[10px] bg-slate-100 px-1.5 py-0.5 rounded text-slate-600">{rootCount}</span>
        )}
      </div>
      
      {expanded && children.length > 0 && (
        <div className="mt-1">
          {children.map(child => (
            <TreeNode 
              key={child.path}
              domain={domain}
              path={child.path}
              name={child.name}
              childrenCount={child.approx_children_count}
              activeDomain={activeDomain}
              activePath={activePath}
              onNavigate={onNavigate}
              level={1}
              maxDepth={maxDepth}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default MemoryTree;
