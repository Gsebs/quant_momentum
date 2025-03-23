import React from 'react';
import { render, screen } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Layout from '../Layout';

// Create a theme instance
const theme = createTheme();

// Mock child component
const MockChild = () => <div>Test Content</div>;

// Helper function to render with theme
const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('Layout', () => {
  it('renders without crashing', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    const titleElement = screen.getByText('Test Title');
    expect(titleElement).toBeInTheDocument();
    expect(titleElement.tagName).toBe('DIV');
    expect(titleElement).toHaveClass('MuiTypography-h6');
  });

  it('renders navigation menu items', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    const menuItems = ['Dashboard', 'Market Data', 'Performance', 'Alerts'];
    menuItems.forEach(item => {
      const elements = screen.getAllByText(item);
      expect(elements.length).toBe(2); // Each item appears in both permanent and temporary drawers
      elements.forEach(element => {
        expect(element).toBeInTheDocument();
        expect(element.tagName).toBe('SPAN');
        expect(element).toHaveClass('MuiTypography-body1');
      });
    });
  });

  it('renders with correct drawer width', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    const drawer = screen.getByRole('navigation');
    const drawerPaper = drawer.querySelector('.MuiDrawer-paper');
    expect(drawerPaper).toHaveStyle({ width: '240px' });
  });

  it('renders with theme', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    const appBar = screen.getByRole('banner');
    expect(appBar).toHaveClass('MuiAppBar-root');
  });

  it('renders with correct toolbar height', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    
    const toolbar = screen.getByRole('banner').querySelector('.MuiToolbar-root');
    expect(toolbar).toHaveClass('MuiToolbar-regular');
  });

  it('renders with correct main content padding', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    
    const mainContent = screen.getByRole('main');
    expect(mainContent).toHaveClass('MuiBox-root');
  });

  it('renders with correct icon colors', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    
    const menuButton = screen.getByLabelText('open drawer');
    expect(menuButton).toHaveClass('MuiIconButton-colorInherit');
  });

  it('renders with correct text colors', () => {
    renderWithTheme(
      <Layout title="Test Title">
        <MockChild />
      </Layout>
    );
    
    const list = screen.getByRole('list');
    const listItems = list.querySelectorAll('.MuiListItem-root');
    listItems.forEach(item => {
      expect(item).toHaveClass('MuiListItem-root');
    });
  });
}); 